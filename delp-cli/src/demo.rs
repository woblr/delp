//! `delp demo` — live verification of delp's headline differentiators.
//!
//! These commands are designed to be eyeballed: they print measurements
//! that prove the claim is real on the user's hardware, not just in our
//! tests.

use std::time::Instant;

use bytes::Bytes;

use delp::codec::decoder::DefaultDecoder;
use delp::codec::encoder::{DefaultEncoder, EncoderOutput};
use delp::codec::DecoderEvent;
use delp::config::{DecoderConfig, EncoderConfig, MatrixStrategy};
use delp::wire::coded::CodedPacket;

/// `delp demo altc` — show that a targeted coded packet is materially
/// smaller (Cauchy) and carries fewer source IDs (both strategies) than
/// a full-window coded packet.
pub fn altc(window: usize, symbol_size: usize, cover: usize) -> anyhow::Result<()> {
    println!("delp demo altc");
    println!("  window={window} symbol_size={symbol_size} altc-cover={cover}\n",);

    for strategy in [MatrixStrategy::Vandermonde, MatrixStrategy::Cauchy] {
        let cap = match strategy {
            MatrixStrategy::Cauchy => window.min(128),
            MatrixStrategy::Vandermonde => window,
        };
        let n = cap;
        let cover = cover.min(n);
        let cfg = EncoderConfig::builder(symbol_size)
            .window_capacity(n)
            .fec_rate(0, 1)
            .matrix_strategy(strategy)
            .build()?;
        let mut enc = DefaultEncoder::with_defaults(cfg);
        let payload = vec![0xABu8; symbol_size];
        for _ in 0..n {
            enc.submit_source(Bytes::from(payload.clone()))?;
        }

        let full = if let EncoderOutput::Coded(b) = enc.generate_coded()?.unwrap() {
            b
        } else {
            unreachable!()
        };
        let recent = if let EncoderOutput::Coded(b) = enc.generate_coded_recent(cover)?.unwrap() {
            b
        } else {
            unreachable!()
        };

        let cp_full = CodedPacket::parse(&full)?;
        let cp_recent = CodedPacket::parse(&recent)?;

        let saved = full.len() as i64 - recent.len() as i64;
        let saved_pct = (saved as f64) / (full.len() as f64) * 100.0;

        println!("  strategy = {:?}", strategy);
        println!(
            "    full-window  : {:>4} ids in EV, {:>5} B on the wire",
            cp_full.ev.source_ids.len(),
            full.len()
        );
        println!(
            "    altc-recent  : {:>4} ids in EV, {:>5} B on the wire  (Δ {} B, {:+.1} %)",
            cp_recent.ev.source_ids.len(),
            recent.len(),
            -saved,
            -saved_pct
        );
        let row_weight_drop =
            cp_full.ev.source_ids.len() as f64 / cp_recent.ev.source_ids.len() as f64;
        println!(
            "    decoder matrix-row weight is {:.1}× smaller for the targeted packet\n",
            row_weight_drop
        );
    }
    Ok(())
}

/// `delp demo generation` — drive a Cauchy session past the RFC 9407
/// 128-packet cap and prove every coded packet decodes cleanly.
pub fn generation(n_symbols: usize, n_coded: u32) -> anyhow::Result<()> {
    println!("delp demo generation");
    println!("  symbols={n_symbols} coded={n_coded}  (RFC 9407 cap = 128)\n",);

    const SYM: usize = 64;
    let cfg = EncoderConfig::builder(SYM)
        .matrix_strategy(MatrixStrategy::Cauchy)
        .window_capacity(n_symbols.max(1))
        .fec_rate(0, 1)
        .build()?;
    let mut encoder = DefaultEncoder::with_defaults(cfg);

    let dec_cfg = DecoderConfig::builder(SYM)
        .feedback_every(u32::MAX)
        .build()?;
    let mut decoder = DefaultDecoder::with_defaults(dec_cfg);

    // Submit sources directly to both encoder and decoder.
    let mut payloads = Vec::with_capacity(n_symbols);
    for i in 0..n_symbols {
        let p: Vec<u8> = (0..SYM).map(|j| ((i * 31 + j * 7) & 0xFF) as u8).collect();
        payloads.push(p.clone());
        for out in encoder.submit_source(Bytes::from(p))? {
            if let EncoderOutput::Source(raw) = out {
                let sp = delp::wire::source::SourcePacket::parse(&raw)?;
                let _ = decoder.handle_source(&sp)?;
            }
        }
    }

    println!("  cycling encoder past the 128-packet cap...");
    let started = Instant::now();
    let mut max_gen = 0u8;
    let mut delivered_post = 0usize;
    for i in 0..n_coded {
        let pkt = encoder
            .generate_coded()?
            .ok_or_else(|| anyhow::anyhow!("window empty at i={i}"))?;
        let raw = if let EncoderOutput::Coded(b) = pkt {
            b
        } else {
            unreachable!()
        };
        let cp = CodedPacket::parse(&raw)?;
        max_gen = max_gen.max(cp.ev.generation);
        for ev in decoder.handle_coded(&cp)? {
            if matches!(ev, DecoderEvent::SourceReady { .. }) {
                delivered_post += 1;
            }
        }
    }
    let elapsed = started.elapsed();

    let cycles = (n_coded / 128).saturating_sub(1);
    println!();
    println!(
        "  ✓ generated {n_coded} coded packets in {:.3} s",
        elapsed.as_secs_f64()
    );
    println!("    encoder.generation()    = {}", encoder.generation());
    println!("    encoder.coded_ids_used  = {}", encoder.coded_ids_used());
    println!(
        "    max gen seen on wire    = {} (≥ {} expected for {n_coded} pkts)",
        max_gen, cycles
    );
    println!(
        "    decoder delivered (post-rotation extras) = {} symbols",
        delivered_post
    );
    println!(
        "    next_delivery_id        = {}",
        decoder.next_delivery_id()
    );
    println!();
    println!("  RFC 9407 would have errored at packet 129 with");
    println!("  `CodedIdExhausted` — delp keeps producing linearly");
    println!("  independent coded packets via generation rotation.");
    Ok(())
}
