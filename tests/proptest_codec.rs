//! Property-based tests for the full encode → channel → decode pipeline.
//!
//! For arbitrary symbol sizes, window sizes, FEC rates, and (deterministic
//! pseudo-random) loss patterns, the codec must either:
//!   - deliver every source symbol byte-exact when redundancy is sufficient, or
//!   - never panic / never deliver a corrupt symbol when it isn't.

#![allow(clippy::too_many_arguments)]

use bytes::Bytes;
use proptest::prelude::*;

use delp::codec::decoder::DefaultDecoder;
use delp::codec::encoder::{DefaultEncoder, EncoderOutput};
use delp::codec::DecoderEvent;
use delp::config::{DecoderConfig, EncoderConfig, Field, MatrixStrategy};
use delp::wire::{coded::CodedPacket, source::SourcePacket};

// ── Repro PRNG ────────────────────────────────────────────────────────────

struct Lcg {
    state: u64,
}
impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }
    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 33) as u32
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

fn run_pipeline(
    field: Field,
    strategy: MatrixStrategy,
    sym_size: usize,
    n_symbols: usize,
    fec_n: usize,
    fec_d: usize,
    drop_pct: u32,
    seed: u64,
) -> (Vec<(u32, Bytes)>, Vec<Vec<u8>>) {
    let enc_cfg = EncoderConfig::builder(sym_size)
        .field(field)
        .matrix_strategy(strategy)
        .window_capacity(n_symbols.max(1))
        .fec_rate(fec_n, fec_d)
        .build()
        .unwrap();
    let dec_cfg = DecoderConfig::builder(sym_size)
        .field(field)
        .feedback_every(u32::MAX)
        .build()
        .unwrap();

    let mut enc = DefaultEncoder::with_defaults(enc_cfg);
    let mut dec = DefaultDecoder::with_defaults(dec_cfg);

    let payloads: Vec<Vec<u8>> = (0..n_symbols)
        .map(|i| {
            (0..sym_size)
                .map(|j| ((i.wrapping_mul(31) + j.wrapping_mul(7)) & 0xFF) as u8)
                .collect()
        })
        .collect();

    let mut wire = Vec::new();
    for p in &payloads {
        for out in enc.submit_source(Bytes::copy_from_slice(p)).unwrap() {
            wire.push(match out {
                EncoderOutput::Source(b) | EncoderOutput::Coded(b) => b,
            });
        }
    }

    let mut rng = Lcg::new(seed);
    let mut delivered = Vec::new();
    for pkt in &wire {
        if rng.next_u32() % 100 < drop_pct {
            continue;
        }
        let events = if pkt[3] == 0x00 {
            let sp = SourcePacket::parse(pkt).unwrap();
            dec.handle_source(&sp).unwrap()
        } else {
            let cp = CodedPacket::parse(pkt).unwrap();
            dec.handle_coded(&cp).unwrap()
        };
        for ev in events {
            if let DecoderEvent::SourceReady { id, data } = ev {
                delivered.push((id, data));
            }
        }
    }
    (delivered, payloads.into_iter().collect())
}

// ── Properties ────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 80, .. ProptestConfig::default()
    })]

    /// Lossless channel: every source symbol must be delivered byte-exact,
    /// in strictly ascending ID order.
    #[test]
    fn lossless_round_trip_delivers_all(
        sym_size  in 16usize..=512,
        n_symbols in 1usize..=32,
        fec_n     in 0usize..=2,
        fec_d     in 1usize..=4,
    ) {
        let (delivered, expected) = run_pipeline(
            Field::Gf2_8,
            MatrixStrategy::Vandermonde,
            sym_size, n_symbols, fec_n, fec_d,
            0, 0,
        );
        prop_assert_eq!(delivered.len(), n_symbols);
        for (i, (id, data)) in delivered.iter().enumerate() {
            prop_assert_eq!(*id as usize, i);
            prop_assert_eq!(data.as_ref(), expected[i].as_slice());
        }
    }

    /// Drop-only-source channel: with 1:1 FEC every source has a coded
    /// counterpart, so all symbols recover regardless of loss pattern.
    #[test]
    fn pure_fec_recovery_with_1to1(
        sym_size  in 16usize..=256,
        n_symbols in 1usize..=24,
        seed      in any::<u64>(),
    ) {
        // Wire format: pkt[3] == 0 → source.  We can't easily express
        // "drop only sources" via run_pipeline(drop_pct), so we encode
        // by-hand: drop every source.
        let enc_cfg = EncoderConfig::builder(sym_size)
            .matrix_strategy(MatrixStrategy::Vandermonde)
            .window_capacity(n_symbols.max(1))
            .fec_rate(1, 1)
            .build().unwrap();
        let dec_cfg = DecoderConfig::builder(sym_size)
            .feedback_every(u32::MAX)
            .build().unwrap();
        let mut enc = DefaultEncoder::with_defaults(enc_cfg);
        let mut dec = DefaultDecoder::with_defaults(dec_cfg);

        let payloads: Vec<Vec<u8>> = (0..n_symbols)
            .map(|i| (0..sym_size).map(|j| ((i + j + seed as usize) & 0xFF) as u8).collect())
            .collect();

        let mut wire = Vec::new();
        for p in &payloads {
            for out in enc.submit_source(Bytes::copy_from_slice(p)).unwrap() {
                wire.push(match out {
                    EncoderOutput::Source(b) | EncoderOutput::Coded(b) => b,
                });
            }
        }

        let mut delivered = Vec::new();
        for pkt in &wire {
            if pkt[3] == 0x00 { continue; } // drop every source
            let cp = CodedPacket::parse(pkt).unwrap();
            for ev in dec.handle_coded(&cp).unwrap() {
                if let DecoderEvent::SourceReady { id, data } = ev {
                    delivered.push((id, data));
                }
            }
        }

        prop_assert_eq!(delivered.len(), n_symbols);
        for (i, (id, data)) in delivered.iter().enumerate() {
            prop_assert_eq!(*id as usize, i);
            prop_assert_eq!(data.as_ref(), payloads[i].as_slice());
        }
    }

    /// Lossy channel: never deliver a corrupt payload.  Recovery may be
    /// partial under heavy loss; whatever IS delivered must be byte-exact
    /// and in ascending order.
    #[test]
    fn never_corrupt_under_random_loss(
        sym_size  in 16usize..=256,
        n_symbols in 1usize..=20,
        drop_pct  in 0u32..=80,
        seed      in any::<u64>(),
    ) {
        let (delivered, expected) = run_pipeline(
            Field::Gf2_8,
            MatrixStrategy::Vandermonde,
            sym_size, n_symbols, 1, 1, drop_pct, seed,
        );
        // Order
        for w in delivered.windows(2) {
            prop_assert!(w[0].0 < w[1].0,
                "out-of-order delivery: {} before {}", w[0].0, w[1].0);
        }
        // Correctness
        for (id, data) in &delivered {
            prop_assert_eq!(data.as_ref(), expected[*id as usize].as_slice(),
                "corrupt payload at id={}", id);
        }
    }

    /// Cauchy strategy: same correctness invariants as Vandermonde.
    #[test]
    fn cauchy_never_corrupt_under_random_loss(
        sym_size  in 16usize..=256,
        n_symbols in 1usize..=8,        // Cauchy GF(2⁸) capacity = 128
        drop_pct  in 0u32..=70,
        seed      in any::<u64>(),
    ) {
        let (delivered, expected) = run_pipeline(
            Field::Gf2_8,
            MatrixStrategy::Cauchy,
            sym_size, n_symbols, 1, 1, drop_pct, seed,
        );
        for w in delivered.windows(2) {
            prop_assert!(w[0].0 < w[1].0);
        }
        for (id, data) in &delivered {
            prop_assert_eq!(data.as_ref(), expected[*id as usize].as_slice());
        }
    }

    /// GF(2⁴) field: same invariants as GF(2⁸) but with the smaller
    /// 14-coded-id session capacity.
    #[test]
    fn gf2_4_never_corrupt(
        sym_size  in 16usize..=128,
        n_symbols in 1usize..=8,
        drop_pct  in 0u32..=60,
        seed      in any::<u64>(),
    ) {
        let (delivered, expected) = run_pipeline(
            Field::Gf2_4,
            MatrixStrategy::Vandermonde,
            sym_size, n_symbols, 1, 1, drop_pct, seed,
        );
        for w in delivered.windows(2) {
            prop_assert!(w[0].0 < w[1].0);
        }
        for (id, data) in &delivered {
            prop_assert_eq!(data.as_ref(), expected[*id as usize].as_slice());
        }
    }
}
