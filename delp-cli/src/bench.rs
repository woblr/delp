//! `delp bench` — single-pass encoder + decoder throughput in MB/s.
//!
//! Generates a deterministic payload, runs it through the codec
//! end-to-end (no network, no loss), and reports wall-clock throughput
//! alongside the SIMD dispatch level chosen on this host.

use std::time::Instant;

use bytes::Bytes;

use delp::codec::decoder::DefaultDecoder;
use delp::codec::encoder::{DefaultEncoder, EncoderOutput};
use delp::codec::DecoderEvent;
use delp::config::{BackpressureMode, DecoderConfig, EncoderConfig};
use delp::wire::{coded::CodedPacket, source::SourcePacket};

use crate::Strategy;

pub fn run(
    symbol_size: usize,
    window: usize,
    n_symbols: usize,
    strategy: Strategy,
) -> anyhow::Result<()> {
    let total_bytes = (symbol_size * n_symbols) as u64;
    println!("delp bench");
    println!(
        "  symbol_size={} window={} symbols={} total={} MiB strategy={:?}",
        symbol_size,
        window,
        n_symbols,
        total_bytes / (1024 * 1024),
        strategy
    );

    // Build encoder.  FEC rate is 1:1 to exercise both the source and
    // coded paths.  Cauchy throughput stops at the (cycle × generations)
    // ceiling: for 1024-byte symbols × 10 000 calls that's well within
    // a `u8` generation counter, so we don't need to cap n_symbols.
    let enc_cfg = EncoderConfig::builder(symbol_size)
        .window_capacity(window)
        .fec_rate(1, 1)
        .matrix_strategy(strategy.into_delp())
        // EvictOldest lets the bench stream past `window` symbols without
        // a real feedback loop — the decoder still receives every coded
        // packet it needs to deliver in order.
        .backpressure(BackpressureMode::EvictOldest)
        .build()?;
    let mut encoder = DefaultEncoder::with_defaults(enc_cfg);

    // Build a matched decoder.
    let dec_cfg = DecoderConfig::builder(symbol_size)
        .feedback_every(u32::MAX)
        .build()?;
    let mut decoder = DefaultDecoder::with_defaults(dec_cfg);

    // Pseudo-random payload (xorshift) — deterministic so successive runs
    // benchmark the same workload.
    let payload: Vec<u8> = {
        let mut x: u64 = 0x9E3779B97F4A7C15;
        (0..symbol_size)
            .map(|_| {
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                (x.wrapping_mul(0x2545F4914F6CDD1D) >> 56) as u8
            })
            .collect()
    };

    // Encode pass.
    let enc_started = Instant::now();
    let mut wire: Vec<Vec<u8>> = Vec::with_capacity(n_symbols * 2);
    for _ in 0..n_symbols {
        let outputs = encoder.submit_source(Bytes::copy_from_slice(&payload))?;
        for out in outputs {
            wire.push(match out {
                EncoderOutput::Source(b) | EncoderOutput::Coded(b) => b,
            });
        }
    }
    let enc_elapsed = enc_started.elapsed();

    // Decode pass.
    let dec_started = Instant::now();
    let mut delivered = 0usize;
    for raw in &wire {
        if raw[3] == 0x00 {
            let sp = SourcePacket::parse(raw)?;
            for ev in decoder.handle_source(&sp)? {
                if matches!(ev, DecoderEvent::SourceReady { .. }) {
                    delivered += 1;
                }
            }
        } else {
            let cp = CodedPacket::parse(raw)?;
            for ev in decoder.handle_coded(&cp)? {
                if matches!(ev, DecoderEvent::SourceReady { .. }) {
                    delivered += 1;
                }
            }
        }
    }
    let dec_elapsed = dec_started.elapsed();

    let enc_mb_s = total_bytes as f64 / 1_000_000.0 / enc_elapsed.as_secs_f64();
    let dec_mb_s = total_bytes as f64 / 1_000_000.0 / dec_elapsed.as_secs_f64();

    println!();
    println!(
        "  encode: {:.2} MB/s  ({} pkts emitted in {:.3}s)",
        enc_mb_s,
        wire.len(),
        enc_elapsed.as_secs_f64()
    );
    println!(
        "  decode: {:.2} MB/s  ({} symbols delivered in {:.3}s)",
        dec_mb_s,
        delivered,
        dec_elapsed.as_secs_f64()
    );

    if delivered != n_symbols {
        anyhow::bail!("decoder delivered {delivered}/{n_symbols} symbols — bench is unsound");
    }
    Ok(())
}
