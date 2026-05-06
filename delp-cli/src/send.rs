//! `delp send` — file sender with optional simulated loss.
//!
//! Wire protocol (on top of delp's coded/source/feedback packets):
//!
//! ```text
//!   START packet (1 datagram, type=0xFF, NOT fed through delp)
//!     [magic: 4 'DELP'][version: 1][reserved: 1][symbol_size: 2 BE]
//!     [n_symbols: 4 BE][file_size: 8 BE][sha256: 32]
//!
//!   delp source + coded packets (the file payload, chunked into symbols)
//!
//!   END packet (1 datagram, type=0xFE) — also outside delp.
//! ```
//!
//! The receiver knows the packet is a control frame because byte 3 of the
//! delp common header is the packet type; control frames use values
//! outside `{0x00, 0x01, 0x02}` so the receiver can demultiplex.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use bytes::Bytes;
use sha2::{Digest, Sha256};
use tokio::net::UdpSocket;
use tokio::time::sleep;

use delp::codec::encoder::{Encoder, EncoderOutput};
use delp::config::{BackpressureMode, EncoderConfig};
use delp::policy::defaults::{AnyAckPolicy, ConstantFecRate, NoCongestionControl};
use delp::policy::ReceiverId;

use crate::{AltcMode, Strategy};

pub struct Config {
    pub file: PathBuf,
    pub dest: SocketAddr,
    pub symbol_size: usize,
    pub window: usize,
    pub fec_n: usize,
    pub fec_d: usize,
    pub strategy: Strategy,
    pub altc: AltcMode,
    pub altc_recent: usize,
    pub loss_rate: f64,
}

pub const CTRL_START: u8 = 0xFF;
pub const CTRL_END: u8 = 0xFE;
const MAGIC: &[u8; 4] = b"DELP";
const PROTO_VERSION: u8 = 1;

pub fn build_start_frame(
    symbol_size: u16,
    n_symbols: u32,
    file_size: u64,
    sha256: &[u8; 32],
) -> Vec<u8> {
    // Layout chosen so byte 3 holds the control type (matches delp's
    // common-header convention).
    let mut buf = Vec::with_capacity(4 + 4 + 4 + 8 + 32);
    buf.extend_from_slice(MAGIC);
    buf.push(PROTO_VERSION);
    buf.push(0); // reserved
    buf.extend_from_slice(&symbol_size.to_be_bytes());
    // Repeat the type tag at byte 3 so the receiver's byte-3 demux sees
    // CTRL_START before the magic check.
    buf[3] = CTRL_START;
    buf.extend_from_slice(&n_symbols.to_be_bytes());
    buf.extend_from_slice(&file_size.to_be_bytes());
    buf.extend_from_slice(sha256);
    buf
}

pub fn build_end_frame() -> Vec<u8> {
    // Minimum 4 bytes so the receiver's byte-3 dispatch works.
    let mut buf = vec![0u8; 4];
    buf[0..4].copy_from_slice(MAGIC);
    buf[3] = CTRL_END;
    buf
}

pub async fn run(cfg: Config) -> anyhow::Result<()> {
    let raw =
        std::fs::read(&cfg.file).with_context(|| format!("reading {}", cfg.file.display()))?;
    let file_size = raw.len() as u64;
    let sha256 = {
        let mut h = Sha256::new();
        h.update(&raw);
        let d = h.finalize();
        let mut out = [0u8; 32];
        out.copy_from_slice(&d);
        out
    };

    // Pad the trailing chunk to a full symbol — the actual file size in
    // the START frame tells the receiver how many bytes to keep.
    let n_symbols = raw.len().div_ceil(cfg.symbol_size);
    let padded_len = n_symbols * cfg.symbol_size;
    let mut padded = raw;
    padded.resize(padded_len, 0);

    println!("delp send");
    println!(
        "  file:        {} ({} bytes)",
        cfg.file.display(),
        file_size
    );
    println!(
        "  destination: {}  symbol_size={} window={} fec={}:{}",
        cfg.dest, cfg.symbol_size, cfg.window, cfg.fec_n, cfg.fec_d
    );
    println!(
        "  strategy:    {:?}  altc={:?}  loss_rate={:.1}%",
        cfg.strategy,
        cfg.altc,
        cfg.loss_rate * 100.0
    );

    // Build encoder.  EvictOldest lets the window slide naturally as
    // the file streams through; the receiver has delivered (or recovered
    // via in-flight coded packets) every evicted symbol by the time the
    // encoder rotates past it.
    let enc_cfg = EncoderConfig::builder(cfg.symbol_size)
        .window_capacity(cfg.window)
        .fec_rate(cfg.fec_n, cfg.fec_d)
        .matrix_strategy(cfg.strategy.into_delp())
        .backpressure(BackpressureMode::EvictOldest)
        .build()?;
    let mut encoder = Encoder::new(
        enc_cfg,
        AnyAckPolicy,
        NoCongestionControl,
        ConstantFecRate::new(cfg.fec_n, cfg.fec_d),
    );

    // Bind a sending socket.
    let socket = Arc::new(UdpSocket::bind("0.0.0.0:0").await?);

    // Loss-simulation PRNG (xorshift64* seeded from the clock).
    let mut rng: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x853c49e6748fea9b);
    if rng == 0 {
        rng = 0x9E3779B97F4A7C15;
    }
    let loss_pct: u32 = (cfg.loss_rate.clamp(0.0, 1.0) * 100.0) as u32;
    let mut should_drop = move || -> bool {
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        ((rng.wrapping_mul(0x2545F4914F6CDD1D) >> 56) as u32 % 100) < loss_pct
    };

    // Fire the START frame (no loss applied — it's metadata).
    let start_frame =
        build_start_frame(cfg.symbol_size as u16, n_symbols as u32, file_size, &sha256);
    socket.send_to(&start_frame, cfg.dest).await?;

    let started = Instant::now();
    let mut sent_src = 0u64;
    let mut sent_cod = 0u64;
    let mut dropped = 0u64;

    // Stream the file as delp source symbols.
    for i in 0..n_symbols {
        let off = i * cfg.symbol_size;
        let chunk = Bytes::copy_from_slice(&padded[off..off + cfg.symbol_size]);
        let outputs = encoder.submit_source(chunk)?;

        // Process each emitted packet.
        for pkt in outputs {
            let (raw, is_source) = match pkt {
                EncoderOutput::Source(b) => (b, true),
                EncoderOutput::Coded(b) => (b, false),
            };
            if should_drop() {
                dropped += 1;
                continue;
            }
            socket.send_to(&raw, cfg.dest).await?;
            if is_source {
                sent_src += 1;
            } else {
                sent_cod += 1;
            }
        }

        // ALTC: emit one extra targeted coded packet per symbol.
        match cfg.altc {
            AltcMode::None => {}
            AltcMode::Recent => {
                if let Some(EncoderOutput::Coded(raw)) =
                    encoder.generate_coded_recent(cfg.altc_recent)?
                {
                    if !should_drop() {
                        socket.send_to(&raw, cfg.dest).await?;
                        sent_cod += 1;
                    } else {
                        dropped += 1;
                    }
                }
            }
            AltcMode::PerReceiver => {
                let rid: ReceiverId = receiver_id_from_addr(cfg.dest);
                if let Some(EncoderOutput::Coded(raw)) = encoder.generate_coded_for_receiver(rid)? {
                    if !should_drop() {
                        socket.send_to(&raw, cfg.dest).await?;
                        sent_cod += 1;
                    } else {
                        dropped += 1;
                    }
                }
            }
        }

        // Pacing: yield every ~32 source submits.  Without this the
        // sender can outrun the receiver's UDP socket buffer on
        // loopback and packets get silently dropped by the kernel —
        // *worse* than the configured loss-rate.
        if i % 32 == 31 {
            sleep(Duration::from_micros(200)).await;
        }
    }

    // Tail flush: after the last source the encoder window still holds
    // up to `window` symbols.  Without extra coded packets, any late
    // source loss in that tail can't be recovered (the window slides
    // off before cod packets cover it).  Flush an extra `window`-sized
    // batch of coded packets so the decoder gets a full FEC budget for
    // the final symbols.
    let flush_count = cfg.window * 2;
    for _ in 0..flush_count {
        match encoder.generate_coded()? {
            Some(EncoderOutput::Coded(raw)) => {
                if !should_drop() {
                    socket.send_to(&raw, cfg.dest).await?;
                    sent_cod += 1;
                } else {
                    dropped += 1;
                }
            }
            _ => break,
        }
    }
    sleep(Duration::from_millis(200)).await;

    // END frame.
    socket.send_to(&build_end_frame(), cfg.dest).await?;

    let elapsed = started.elapsed();
    let mb_per_s = file_size as f64 / 1_000_000.0 / elapsed.as_secs_f64();
    println!();
    println!(
        "✓ sent {file_size} bytes in {:.3} s ({:.2} MB/s)",
        elapsed.as_secs_f64(),
        mb_per_s
    );
    println!("  source pkts={sent_src}  coded pkts={sent_cod}  dropped(simulated)={dropped}");
    println!("  encoder generation={}", encoder.generation());

    Ok(())
}

fn receiver_id_from_addr(addr: SocketAddr) -> ReceiverId {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    addr.hash(&mut h);
    h.finish()
}
