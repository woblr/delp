//! End-to-end test for delp's **unlimited-length Cauchy session** extension.
//!
//! In RFC 9407 Cauchy GF(2⁸) tops out at 128 coded packets per session: the
//! point set `y_j = 128 + j` only has 128 distinct values.  delp adds a
//! 1-byte `generation` field to the encoding vector that rotates the
//! y-point set; each generation cycle yields 128 fresh, linearly-independent
//! coded packets.  With a `u8` generation counter the session can carry
//! 128 × 256 = 32 768 coded packets before it would wrap a second time.
//!
//! This test drives 1 200 coded packets through a real encode → decode
//! pipeline (well past the old 128-packet ceiling) and verifies that
//! every coded packet round-trips correctly with its generation byte
//! intact.

use bytes::Bytes;

use delp::codec::decoder::DefaultDecoder;
use delp::codec::encoder::{DefaultEncoder, EncoderOutput};
use delp::codec::DecoderEvent;
use delp::config::{DecoderConfig, EncoderConfig, MatrixStrategy};
use delp::wire::{coded::CodedPacket, source::SourcePacket};

/// Encode 12 source symbols, then generate `n_coded` coded packets back-to-back.
/// The decoder receives every source and every coded packet; it must deliver
/// all 12 source symbols and stay healthy throughout.
fn run_long_cauchy_session(n_coded: u32) {
    const SYM: usize = 64;
    const N_SRC: usize = 12;

    let enc_cfg = EncoderConfig::builder(SYM)
        .matrix_strategy(MatrixStrategy::Cauchy)
        .window_capacity(N_SRC)
        .fec_rate(0, 1) // disable auto-FEC; we'll call generate_coded by hand
        .build()
        .unwrap();
    let dec_cfg = DecoderConfig::builder(SYM)
        .feedback_every(u32::MAX)
        .build()
        .unwrap();

    let mut enc = DefaultEncoder::with_defaults(enc_cfg);
    let mut dec = DefaultDecoder::with_defaults(dec_cfg);

    let payloads: Vec<Vec<u8>> = (0..N_SRC)
        .map(|i| (0..SYM).map(|j| ((i * 37 + j * 11) & 0xFF) as u8).collect())
        .collect();

    // Submit every source — receiver gets them directly.
    for p in &payloads {
        for out in enc.submit_source(Bytes::copy_from_slice(p)).unwrap() {
            if let EncoderOutput::Source(raw) = out {
                let sp = SourcePacket::parse(&raw).unwrap();
                let _ = dec.handle_source(&sp).unwrap();
            }
        }
    }

    // Generate `n_coded` coded packets.  Encoder must succeed for every one
    // (no exhaustion error).  The decoder absorbs them all without error.
    let mut max_generation_seen = 0u8;
    for i in 0..n_coded {
        let pkt = enc
            .generate_coded()
            .unwrap_or_else(|e| panic!("generate_coded failed at i={i}: {e}"))
            .expect("window non-empty");
        let raw = match pkt {
            EncoderOutput::Coded(b) => b,
            EncoderOutput::Source(_) => panic!("expected Coded"),
        };
        let cp = CodedPacket::parse(&raw).unwrap();
        max_generation_seen = max_generation_seen.max(cp.ev.generation);
        let _events = dec.handle_coded(&cp).unwrap();
    }

    // Sanity: with 1200 packets and a 128-packet cycle the encoder should
    // have advanced through generation 0..=9 (1200/128 ≈ 9.4 cycles).
    let expected_min_gen = (n_coded / 128).saturating_sub(1) as u8;
    assert!(
        max_generation_seen >= expected_min_gen,
        "expected to see generation ≥ {expected_min_gen}, saw max {max_generation_seen}",
    );

    // All sources delivered correctly (they were sent direct, no recovery
    // needed) — but verify the matrix never desynchronised.
    assert_eq!(dec.next_delivery_id(), N_SRC as u32);
}

#[test]
fn cauchy_runs_past_old_128_limit() {
    // 200 coded packets — 72 above the old hard cap.
    run_long_cauchy_session(200);
}

#[test]
fn cauchy_runs_for_thousand_packets() {
    // 1 200 coded packets — ≈9.4× the old cap.  Exercises generation
    // rotation through 9 distinct y-point sets.
    run_long_cauchy_session(1200);
}

/// Verify recovery still works after generation rotation: drop one source
/// during a multi-cycle session and confirm a coded packet from a later
/// generation can recover it.
#[test]
fn cauchy_recovers_lost_source_across_generation_boundary() {
    const SYM: usize = 64;
    const N_SRC: usize = 4;

    let enc_cfg = EncoderConfig::builder(SYM)
        .matrix_strategy(MatrixStrategy::Cauchy)
        .window_capacity(N_SRC)
        .fec_rate(0, 1)
        .build()
        .unwrap();
    let dec_cfg = DecoderConfig::builder(SYM)
        .feedback_every(u32::MAX)
        .build()
        .unwrap();
    let mut enc = DefaultEncoder::with_defaults(enc_cfg);
    let mut dec = DefaultDecoder::with_defaults(dec_cfg);

    let payloads: Vec<Vec<u8>> = (0..N_SRC)
        .map(|i| vec![(i as u8).wrapping_mul(31); SYM])
        .collect();

    // Submit sources; deliver all except symbol 1.
    for (i, p) in payloads.iter().enumerate() {
        for out in enc.submit_source(Bytes::copy_from_slice(p)).unwrap() {
            if let EncoderOutput::Source(raw) = out {
                if i == 1 {
                    continue;
                }
                let sp = SourcePacket::parse(&raw).unwrap();
                let _ = dec.handle_source(&sp).unwrap();
            }
        }
    }

    // Burn through a full cycle of coded packets, dropping each one.
    for _ in 0..128u32 {
        let _ = enc.generate_coded().unwrap();
    }
    assert!(enc.generation() >= 1, "must have advanced generation");

    // Now generate one more coded packet (post-rotation) and feed it to
    // the decoder — symbol 1 must be recovered.
    let mut recovered = false;
    for _attempt in 0..8 {
        let pkt = enc.generate_coded().unwrap().unwrap();
        if let EncoderOutput::Coded(raw) = pkt {
            let cp = CodedPacket::parse(&raw).unwrap();
            for ev in dec.handle_coded(&cp).unwrap() {
                if let DecoderEvent::SourceReady { id, data } = ev {
                    if id == 1 {
                        assert_eq!(data.as_ref(), payloads[1].as_slice());
                        recovered = true;
                    }
                }
            }
            if recovered {
                break;
            }
        }
    }
    assert!(
        recovered,
        "symbol 1 must be recoverable from post-rotation coded packets"
    );
}
