//! Integration tests for **Adaptive Loss-Targeted Coding** (ALTC).
//!
//! ALTC is delp's extension that lets the encoder generate coded packets
//! covering only a *subset* of the encoding window — the symbols a specific
//! receiver still needs, the most-recent N symbols, or any custom subset.
//!
//! Compared with the standard "cover everything" rule, ALTC:
//!   1. shrinks the encoding-vector wire size when only a tail of the
//!      window matters (smaller `nb_ids`, denser id-storage compression);
//!   2. cuts the encoder's GF-multiplication count proportionally to
//!      the cover size;
//!   3. lowers the decoder's matrix-row weight, which keeps Gaussian
//!      elimination fast even with very large windows.
//!
//! Tests verify functional correctness AND quantify the wire-size /
//! row-weight advantage.

use bytes::Bytes;

use delp::codec::decoder::DefaultDecoder;
use delp::codec::encoder::{DefaultEncoder, EncoderOutput};
use delp::codec::DecoderEvent;
use delp::config::{DecoderConfig, EncoderConfig, MatrixStrategy};
use delp::wire::{coded::CodedPacket, source::SourcePacket};

const SYM: usize = 64;
const N_SRC: usize = 32;

fn build_pair() -> (DefaultEncoder, DefaultDecoder, Vec<Vec<u8>>) {
    let enc_cfg = EncoderConfig::builder(SYM)
        .matrix_strategy(MatrixStrategy::Vandermonde)
        .window_capacity(N_SRC)
        .fec_rate(0, 1)
        .build()
        .unwrap();
    let dec_cfg = DecoderConfig::builder(SYM)
        .feedback_every(u32::MAX)
        .build()
        .unwrap();
    let mut enc = DefaultEncoder::with_defaults(enc_cfg);
    let dec = DefaultDecoder::with_defaults(dec_cfg);
    let payloads: Vec<Vec<u8>> = (0..N_SRC)
        .map(|i| (0..SYM).map(|j| ((i * 17 + j * 5) & 0xFF) as u8).collect())
        .collect();
    for p in &payloads {
        let _ = enc.submit_source(Bytes::copy_from_slice(p)).unwrap();
    }
    (enc, dec, payloads)
}

/// Targeted coverage with **Cauchy** (explicit coefficients): the EV
/// coefficient field shrinks linearly with the cover size, so the wire
/// packet is strictly smaller than a full-window coded packet.
#[test]
fn targeted_cauchy_packet_is_smaller_than_full_window() {
    let enc_cfg = EncoderConfig::builder(SYM)
        .matrix_strategy(MatrixStrategy::Cauchy)
        .window_capacity(N_SRC)
        .fec_rate(0, 1)
        .build()
        .unwrap();
    let mut enc = DefaultEncoder::with_defaults(enc_cfg);
    for i in 0..N_SRC {
        let p: Vec<u8> = (0..SYM).map(|j| ((i * 17 + j * 5) & 0xFF) as u8).collect();
        let _ = enc.submit_source(Bytes::from(p)).unwrap();
    }

    let cover_recent: Vec<u32> = (N_SRC as u32 - 4..N_SRC as u32).collect();
    let targeted = match enc.generate_coded_targeted(&cover_recent).unwrap().unwrap() {
        EncoderOutput::Coded(b) => b,
        _ => panic!(),
    };
    let full = match enc.generate_coded().unwrap().unwrap() {
        EncoderOutput::Coded(b) => b,
        _ => panic!(),
    };
    assert!(
        targeted.len() < full.len(),
        "Cauchy targeted={} bytes, full={} bytes — targeted should be smaller",
        targeted.len(),
        full.len()
    );

    let cp_t = CodedPacket::parse(&targeted).unwrap();
    let cp_f = CodedPacket::parse(&full).unwrap();
    assert_eq!(cp_t.ev.source_ids.len(), 4);
    assert_eq!(cp_f.ev.source_ids.len(), N_SRC);
    assert_eq!(
        cp_t.ev.coefficients.len(),
        4,
        "Cauchy uses explicit coefs — count must match cover size"
    );
    assert_eq!(cp_f.ev.coefficients.len(), N_SRC);
}

/// Vandermonde targeted: regardless of id-storage encoding, the EV's
/// `nb_ids` count is strictly smaller than full-window — which directly
/// translates to fewer GF mul operations in the encoder and a sparser
/// matrix row in the decoder.  (Wire-format size depends on id contiguity
/// and may be equal or slightly larger when the cover is non-contiguous.)
#[test]
fn targeted_vandermonde_reduces_matrix_row_weight() {
    let (mut enc, _, _) = build_pair();

    let cover: Vec<u32> = vec![1, 5, 9, 13, 17, 21];
    let targeted = match enc.generate_coded_targeted(&cover).unwrap().unwrap() {
        EncoderOutput::Coded(b) => b,
        _ => panic!(),
    };
    let full = match enc.generate_coded().unwrap().unwrap() {
        EncoderOutput::Coded(b) => b,
        _ => panic!(),
    };

    let cp_t = CodedPacket::parse(&targeted).unwrap();
    let cp_f = CodedPacket::parse(&full).unwrap();
    assert_eq!(cp_t.ev.source_ids.len(), 6);
    assert_eq!(cp_f.ev.source_ids.len(), N_SRC);
    assert!(
        cp_t.ev.source_ids.len() < cp_f.ev.source_ids.len(),
        "ALTC targeted EV must list fewer source IDs ({}) than full ({})",
        cp_t.ev.source_ids.len(),
        cp_f.ev.source_ids.len()
    );
}

/// `generate_coded_recent(n)` selects exactly the n most-recently-submitted
/// source IDs.
#[test]
fn recent_selects_tail() {
    let (mut enc, _dec, _payloads) = build_pair();

    let pkt = match enc.generate_coded_recent(8).unwrap().unwrap() {
        EncoderOutput::Coded(b) => b,
        _ => panic!(),
    };
    let cp = CodedPacket::parse(&pkt).unwrap();
    let expected: Vec<u32> = (N_SRC as u32 - 8..N_SRC as u32).collect();
    assert_eq!(cp.ev.source_ids.as_slice(), expected.as_slice());
}

/// Recovery from a targeted coded packet: drop the targeted source and
/// confirm the targeted coded packet alone (combined with surviving direct
/// sources) recovers it.
#[test]
fn targeted_packet_recovers_lost_source() {
    let (mut enc, mut dec, payloads) = build_pair();

    // Deliver every source EXCEPT id=20 (one of the recent-tail symbols).
    for (i, p) in payloads.iter().enumerate() {
        if i == 20 {
            continue;
        }
        let raw = SourcePacket::serialise(i as u32, p, &[], None);
        let sp = SourcePacket::parse(&raw).unwrap();
        let _ = dec.handle_source(&sp).unwrap();
    }

    // Generate a coded packet that targets the recent-loss tail (16..32).
    let cover: Vec<u32> = (16u32..32).collect();
    let pkt = match enc.generate_coded_targeted(&cover).unwrap().unwrap() {
        EncoderOutput::Coded(b) => b,
        _ => panic!(),
    };
    let cp = CodedPacket::parse(&pkt).unwrap();
    assert_eq!(cp.ev.source_ids.len(), 16, "EV must list exactly the cover");

    let mut recovered_correct = false;
    for ev in dec.handle_coded(&cp).unwrap() {
        if let DecoderEvent::SourceReady { id, data } = ev {
            if id == 20 {
                assert_eq!(
                    data.as_ref(),
                    payloads[20].as_slice(),
                    "ALTC recovered symbol 20 with correct payload"
                );
                recovered_correct = true;
            }
        }
    }
    assert!(recovered_correct, "symbol 20 should have been recovered");
}

/// `generate_coded_for_receiver` drops symbols already ACK'd by that
/// receiver, producing a smaller coded packet tailored to the laggard.
#[test]
fn for_receiver_excludes_acked_symbols() {
    use delp::wire::feedback::FeedbackPacket;

    let (mut enc, _dec, _payloads) = build_pair();

    // Receiver 1 ACK'd symbols 0..16; symbols 16..32 are still pending.
    let acked: Vec<u32> = (0..16u32).collect();
    let fb = FeedbackPacket::build(0, 0, 0, 0.0, &acked);
    enc.handle_feedback(1, &fb);

    // The window may have been evicted by AnyAckPolicy — refill if so.
    // Use a fresh encoder with AllAckPolicy for this test instead.
    let enc_cfg = EncoderConfig::builder(SYM)
        .matrix_strategy(MatrixStrategy::Vandermonde)
        .window_capacity(N_SRC)
        .fec_rate(0, 1)
        .build()
        .unwrap();
    let mut enc2 = delp::codec::encoder::Encoder::new(
        enc_cfg,
        delp::policy::defaults::AllAckPolicy,
        delp::policy::defaults::NoCongestionControl,
        delp::policy::defaults::ConstantFecRate::disabled(),
    );
    for i in 0..N_SRC {
        let p: Vec<u8> = (0..SYM).map(|j| ((i * 17 + j * 5) & 0xFF) as u8).collect();
        let _ = enc2.submit_source(Bytes::from(p)).unwrap();
    }

    // Receiver 1 ACKs first 16 symbols.
    let fb = FeedbackPacket::build(0, 0, 0, 0.0, &(0..16u32).collect::<Vec<_>>());
    enc2.handle_feedback(1, &fb);

    // generate_coded_for_receiver(1) should cover only 16..32.
    let pkt = match enc2.generate_coded_for_receiver(1).unwrap().unwrap() {
        EncoderOutput::Coded(b) => b,
        _ => panic!(),
    };
    let cp = CodedPacket::parse(&pkt).unwrap();
    let cover: Vec<u32> = cp.ev.source_ids.to_vec();
    assert_eq!(cover.len(), 16, "should cover only the unacknowledged tail");
    assert!(
        cover.iter().all(|&id| id >= 16),
        "covered IDs must all be ≥ 16, got {cover:?}"
    );
}

/// `generate_coded_for_receiver` returns Ok(None) when the receiver has
/// already acknowledged every windowed symbol.
#[test]
fn for_receiver_returns_none_when_fully_acked() {
    use delp::wire::feedback::FeedbackPacket;

    let enc_cfg = EncoderConfig::builder(SYM)
        .matrix_strategy(MatrixStrategy::Vandermonde)
        .window_capacity(N_SRC)
        .fec_rate(0, 1)
        .build()
        .unwrap();
    let mut enc = delp::codec::encoder::Encoder::new(
        enc_cfg,
        delp::policy::defaults::AllAckPolicy,
        delp::policy::defaults::NoCongestionControl,
        delp::policy::defaults::ConstantFecRate::disabled(),
    );
    for i in 0..N_SRC {
        let p: Vec<u8> = vec![i as u8; SYM];
        let _ = enc.submit_source(Bytes::from(p)).unwrap();
    }

    // Receiver acks every symbol in window.
    let all: Vec<u32> = (0..N_SRC as u32).collect();
    let fb = FeedbackPacket::build(0, 0, 0, 0.0, &all);
    enc.handle_feedback(7, &fb);

    let result = enc.generate_coded_for_receiver(7).unwrap();
    assert!(
        result.is_none(),
        "should return None when receiver already has every windowed symbol"
    );
}

/// Empty cover set is silently treated as "nothing to do" rather than an error.
#[test]
fn targeted_with_empty_cover_returns_none() {
    let (mut enc, _, _) = build_pair();
    let result = enc.generate_coded_targeted(&[]).unwrap();
    assert!(result.is_none());
}

/// IDs outside the window are filtered out.
#[test]
fn targeted_filters_ids_not_in_window() {
    let (mut enc, _, _) = build_pair();
    // Mix valid (in window) and invalid IDs.
    let cover: Vec<u32> = vec![5, 6, 99_999, 8, 1_000_000];
    let pkt = enc.generate_coded_targeted(&cover).unwrap().unwrap();
    let raw = match pkt {
        EncoderOutput::Coded(b) => b,
        _ => panic!(),
    };
    let cp = CodedPacket::parse(&raw).unwrap();
    assert_eq!(
        cp.ev.source_ids.as_slice(),
        &[5u32, 6, 8],
        "out-of-window IDs must be filtered"
    );
}
