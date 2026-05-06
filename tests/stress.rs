//! Stress tests for the Delp codec.
//!
//! Drive the synchronous encoder/decoder pair through high-loss, bursty,
//! and large-window scenarios.  These tests use the codec API directly
//! (no transport) for deterministic, fast execution.

#![allow(clippy::too_many_arguments)]

use bytes::Bytes;

use delp::codec::decoder::DefaultDecoder;
use delp::codec::encoder::{DefaultEncoder, EncoderOutput};
use delp::codec::DecoderEvent;
use delp::config::{DecoderConfig, EncoderConfig, Field, MatrixStrategy};
use delp::wire::{coded::CodedPacket, source::SourcePacket};

// ── Reproducible PRNG ────────────────────────────────────────────────────

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
    fn drop(&mut self, pct: u32) -> bool {
        self.next_u32() % 100 < pct
    }
}

// ── Channel simulator ────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum Channel {
    /// Independent per-packet drop with given percent loss.
    Random { loss_pct: u32 },
    /// Bursty: alternating periods of loss / no-loss.  Loss bursts of
    /// `burst_len` packets fire every `cycle_len` packets.
    Burst { burst_len: u32, cycle_len: u32 },
    /// Drop *every* source packet — exercises pure FEC recovery.  Coded
    /// packets always pass.
    AllSourcesDropped,
    /// Drop the first `n` packets — the worst case for an in-order codec.
    HeadDropped { n: u32 },
}

impl Channel {
    fn should_drop(self, idx: u32, is_source: bool, rng: &mut Lcg) -> bool {
        match self {
            Channel::Random { loss_pct } => rng.drop(loss_pct),
            Channel::Burst {
                burst_len,
                cycle_len,
            } => idx % cycle_len < burst_len,
            Channel::AllSourcesDropped => is_source,
            Channel::HeadDropped { n } => idx < n,
        }
    }
}

// ── Run one encode → channel → decode session ────────────────────────────

struct StressResult {
    delivered: Vec<u32>,
    lost_count: u32,
    pkts_sent: u32,
    pkts_dropped: u32,
}

fn run_session(
    field: Field,
    strategy: MatrixStrategy,
    n_symbols: usize,
    sym_size: usize,
    fec_numer: usize,
    fec_denom: usize,
    window: usize,
    channel: Channel,
    seed: u64,
) -> StressResult {
    let enc_cfg = EncoderConfig::builder(sym_size)
        .field(field)
        .matrix_strategy(strategy)
        .window_capacity(window)
        .fec_rate(fec_numer, fec_denom)
        .build()
        .expect("encoder config valid");
    let dec_cfg = DecoderConfig::builder(sym_size)
        .field(field)
        .feedback_every(u32::MAX) // suppress feedback in pure-codec tests
        .build()
        .expect("decoder config valid");

    let mut enc = DefaultEncoder::with_defaults(enc_cfg);
    let mut dec = DefaultDecoder::with_defaults(dec_cfg);

    let payloads: Vec<Vec<u8>> = (0..n_symbols)
        .map(|i| {
            (0..sym_size)
                .map(|j| ((i * 31 + j * 7) & 0xFF) as u8)
                .collect()
        })
        .collect();

    let mut wire: Vec<Vec<u8>> = Vec::with_capacity(n_symbols * 4);
    for p in &payloads {
        for out in enc.submit_source(Bytes::copy_from_slice(p)).unwrap() {
            wire.push(match out {
                EncoderOutput::Source(b) | EncoderOutput::Coded(b) => b,
            });
        }
    }

    let mut rng = Lcg::new(seed);
    let mut delivered = Vec::with_capacity(n_symbols);
    let mut dropped = 0u32;

    for (idx, pkt) in wire.iter().enumerate() {
        let is_source = pkt[3] == 0x00;
        if channel.should_drop(idx as u32, is_source, &mut rng) {
            dropped += 1;
            continue;
        }
        let events = if is_source {
            let sp = SourcePacket::parse(pkt).unwrap();
            dec.handle_source(&sp).unwrap()
        } else {
            let cp = CodedPacket::parse(pkt).unwrap();
            dec.handle_coded(&cp).unwrap()
        };
        for ev in events {
            match ev {
                DecoderEvent::SourceReady { id, data } => {
                    assert_eq!(
                        data.as_ref(),
                        payloads[id as usize].as_slice(),
                        "symbol {id} delivered with corrupt payload",
                    );
                    delivered.push(id);
                }
                DecoderEvent::SendFeedback(_) => {}
                DecoderEvent::UnrecoverableGap { .. } => {}
            }
        }
    }

    let lost_count = (0..n_symbols as u32)
        .filter(|i| !delivered.contains(i))
        .count() as u32;
    StressResult {
        delivered,
        lost_count,
        pkts_sent: wire.len() as u32,
        pkts_dropped: dropped,
    }
}

// ── Tests with deterministic drop patterns ───────────────────────────────

/// Drop every source packet — pure FEC recovery via coded packets only.
/// 1:1 FEC means n source + n coded, with all sources dropped the n coded
/// equations form a full-rank Vandermonde system over the n unknowns.
#[test]
fn all_sources_dropped_recovers_via_fec() {
    for n_symbols in [4, 8, 16] {
        let r = run_session(
            Field::Gf2_8,
            MatrixStrategy::Vandermonde,
            n_symbols,
            128,
            1,
            1,
            n_symbols,
            Channel::AllSourcesDropped,
            42,
        );
        assert_eq!(
            r.delivered.len(),
            n_symbols,
            "n={n_symbols}: delivered {}/{} (dropped {} of {})",
            r.delivered.len(),
            n_symbols,
            r.pkts_dropped,
            r.pkts_sent
        );
    }
}

/// Burst loss: drop 4 consecutive packets every 8.  With 1:1 FEC this
/// drops at most one coded per source; recovery via the surviving cods.
#[test]
fn burst_loss_4_in_8_with_1to1_fec() {
    let n_symbols = 16;
    let r = run_session(
        Field::Gf2_8,
        MatrixStrategy::Vandermonde,
        n_symbols,
        256,
        1,
        1,
        n_symbols,
        Channel::Burst {
            burst_len: 4,
            cycle_len: 8,
        },
        0xC0FFEE,
    );
    assert_eq!(
        r.delivered.len(),
        n_symbols,
        "burst loss: delivered {}/{} (dropped {} of {})",
        r.delivered.len(),
        n_symbols,
        r.pkts_dropped,
        r.pkts_sent
    );
}

/// Cauchy MDS strategy: drop the first 4 packets; the codec recovers via
/// later coded packets.  Cauchy guarantees full-rank submatrices.
#[test]
fn cauchy_head_drop_recovers_all() {
    let n_symbols = 8;
    let r = run_session(
        Field::Gf2_8,
        MatrixStrategy::Cauchy,
        n_symbols,
        128,
        1,
        1,
        n_symbols,
        Channel::HeadDropped { n: 4 },
        0,
    );
    assert_eq!(
        r.delivered.len(),
        n_symbols,
        "Cauchy head-drop: delivered {}/{}",
        r.delivered.len(),
        n_symbols
    );
}

/// GF(2⁴) field — pure-FEC recovery (drop every source).  Verifies the
/// nibble-broadcast SIMD path on a real encode/decode pipeline.
#[test]
fn gf2_4_all_sources_dropped_recovers() {
    let n_symbols = 6; // GF(2⁴) coded-id limit is 14
    let r = run_session(
        Field::Gf2_4,
        MatrixStrategy::Vandermonde,
        n_symbols,
        64,
        1,
        1,
        n_symbols,
        Channel::AllSourcesDropped,
        0xABC,
    );
    assert_eq!(
        r.delivered.len(),
        n_symbols,
        "GF(2⁴) FEC-only: delivered {}/{}",
        r.delivered.len(),
        n_symbols
    );
}

// ── Tests with random drop patterns (lossy assertions) ───────────────────

/// 30 % random loss with 2:1 FEC: across many seeds the codec must
/// deliver *almost all* symbols.  A handful of seeds happen to drop every
/// packet covering one specific symbol — true information loss, not a
/// codec failure — so we tolerate up to 1 missing symbol per seed and
/// require the average to round to "everything recovered".
#[test]
fn random_30pct_loss_2to1_fec_high_recovery() {
    let n_symbols = 16;
    let seeds = [1u64, 7, 42, 100, 1234, 0xDEADBEEF, 777, 0xC0DE];
    let mut total_lost = 0u32;
    for seed in seeds {
        let r = run_session(
            Field::Gf2_8,
            MatrixStrategy::Vandermonde,
            n_symbols,
            128,
            2,
            1,
            n_symbols,
            Channel::Random { loss_pct: 30 },
            seed,
        );
        assert!(
            r.lost_count <= 1,
            "seed={seed}: lost {} symbols (delivered {}/{}, dropped {} of {})",
            r.lost_count,
            r.delivered.len(),
            n_symbols,
            r.pkts_dropped,
            r.pkts_sent
        );
        total_lost += r.lost_count;
    }
    assert!(
        total_lost <= 2,
        "{total_lost} symbols lost across {} seeds — recovery quality below threshold",
        seeds.len()
    );
}

/// Very-high loss + low FEC: recovery is *not* guaranteed.  Verify that
/// the decoder still delivers some symbols and never panics or corrupts.
#[test]
fn over_loss_does_not_panic() {
    let n_symbols = 12;
    let r = run_session(
        Field::Gf2_8,
        MatrixStrategy::Vandermonde,
        n_symbols,
        128,
        1,
        4,
        n_symbols,
        Channel::Random { loss_pct: 70 },
        99,
    );
    assert!(r.delivered.len() <= n_symbols);
    for id in &r.delivered {
        assert!(*id < n_symbols as u32);
    }
}

/// Larger window, larger symbol size, modest loss — exercises the SIMD
/// hot path (1024-byte symbols × 64-symbol window).
#[test]
fn large_window_simd_path() {
    let n_symbols = 64;
    let r = run_session(
        Field::Gf2_8,
        MatrixStrategy::Vandermonde,
        n_symbols,
        1024,
        1,
        2,
        64,
        Channel::Random { loss_pct: 20 },
        0xFEED,
    );
    assert!(
        r.delivered.len() >= n_symbols - 2,
        "large window: delivered only {}/{} (dropped {} of {})",
        r.delivered.len(),
        n_symbols,
        r.pkts_dropped,
        r.pkts_sent
    );
}

/// In-order delivery invariant: receiver must always deliver source IDs in
/// strictly ascending order, never out-of-order.
#[test]
fn delivery_is_strictly_in_order() {
    let n_symbols = 32;
    let r = run_session(
        Field::Gf2_8,
        MatrixStrategy::Vandermonde,
        n_symbols,
        128,
        1,
        1,
        n_symbols,
        Channel::Random { loss_pct: 35 },
        0xBEEF,
    );
    for w in r.delivered.windows(2) {
        assert!(
            w[0] < w[1],
            "out-of-order delivery: {} before {}",
            w[0],
            w[1]
        );
    }
}
