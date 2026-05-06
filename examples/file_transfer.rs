//! End-to-end file transfer over a lossy UDP link, using `DelpSession`.
//!
//! The sender chunks a synthetic 256 KB blob into fixed-size symbols and
//! pushes them through a `DelpSession`.  A simulated proxy randomly drops
//! 25 % of source packets (coded packets always pass).  The receiver
//! reconstructs the blob and verifies a byte-exact match.
//!
//! Demonstrates:
//!   - `DelpSession::split` for concurrent send/receive halves
//!   - `SessionBuilder` for one-line bidirectional setup
//!   - Realistic chunked transfer of a multi-KB payload
//!
//! Usage:
//!   cargo run --example file_transfer --features async

#[cfg(not(feature = "async"))]
fn main() {
    eprintln!("This example requires --features async");
    std::process::exit(1);
}

#[cfg(feature = "async")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use bytes::Bytes;
    use tokio::net::UdpSocket;
    use tokio::time::sleep;

    use delp::config::{DecoderConfig, EncoderConfig, MatrixStrategy};
    use delp::transport::{FecReceiver, FecSender};
    use delp::{DefaultDecoder, DefaultEncoder};

    // ── Configuration ─────────────────────────────────────────────────────

    const SYMBOL_SIZE: usize = 1024; // 1 KB per symbol
    const BLOB_BYTES: usize = 128 * 1024; // 128 KB blob → 128 symbols
    const WINDOW: usize = 32; // sliding window
    const LOSS_PCT: u8 = 25; // simulated source-packet loss

    // Vandermonde GF(2⁸) supports 254 unique coded IDs per session;
    // 128 sources × 1:1 FEC → 128 codeds, well within the limit.

    let n_symbols = BLOB_BYTES / SYMBOL_SIZE;

    // ── Synthesise the source blob ────────────────────────────────────────
    //
    // A pseudo-random byte stream — non-trivial enough that any
    // reconstruction error is visible in the byte-equality check below.

    let blob: Vec<u8> = {
        let mut state: u64 = 0xABCDEF1234567890;
        (0..BLOB_BYTES)
            .map(|_| {
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                (state.wrapping_mul(0x2545f4914f6cdd1d) >> 56) as u8
            })
            .collect()
    };

    // ── Sockets ───────────────────────────────────────────────────────────
    // sender_sock  ──► [proxy applies loss] ──► receiver_sock
    //      ◄─────────────── feedback ───────────────────

    let sender_sock = Arc::new(UdpSocket::bind("127.0.0.1:0").await?);
    let proxy_sock = Arc::new(UdpSocket::bind("127.0.0.1:0").await?);
    let receiver_sock = Arc::new(UdpSocket::bind("127.0.0.1:0").await?);

    let sender_addr = sender_sock.local_addr()?;
    let proxy_addr = proxy_sock.local_addr()?;
    let receiver_addr = receiver_sock.local_addr()?;

    println!("File transfer demo");
    println!("  blob:    {BLOB_BYTES} bytes ({n_symbols} symbols × {SYMBOL_SIZE} B)");
    println!("  window:  {WINDOW} symbols");
    println!("  loss:    {LOSS_PCT} % source-packet drop (coded always forwarded)");
    println!();

    // ── Encoder / decoder pair ────────────────────────────────────────────
    //
    // 1:1 FEC ratio → for every source packet the encoder emits one coded
    // packet.  Plenty of redundancy to cover 25 % loss.

    let enc_cfg = EncoderConfig::builder(SYMBOL_SIZE)
        .window_capacity(WINDOW)
        .fec_rate(1, 1)
        .matrix_strategy(MatrixStrategy::Vandermonde)
        .build()?;
    let dec_cfg = DecoderConfig::builder(SYMBOL_SIZE)
        .feedback_every(8)
        .build()?;

    let encoder = DefaultEncoder::with_defaults(enc_cfg);
    let decoder = DefaultDecoder::with_defaults(dec_cfg);

    let mut sender = FecSender::new(encoder, Arc::clone(&sender_sock), proxy_addr);
    let mut receiver = FecReceiver::new(decoder, Arc::clone(&receiver_sock), sender_addr);

    // ── Lossy proxy task ──────────────────────────────────────────────────

    let proxy = {
        let psock = Arc::clone(&proxy_sock);
        let rx_addr = receiver_addr;
        let tx_addr = sender_addr;
        tokio::spawn(async move {
            let mut buf = vec![0u8; 65536];
            let mut state = 0xC0FFEEu64;
            let mut dropped = 0u32;
            let mut forwarded = 0u32;
            loop {
                let (n, _) = psock.recv_from(&mut buf).await.unwrap();
                let raw = buf[..n].to_vec();

                // Feedback (0x02) — always forward back to sender.
                if raw.len() >= 4 && raw[3] == 0x02 {
                    psock.send_to(&raw, tx_addr).await.unwrap();
                    continue;
                }

                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                let r = (state.wrapping_mul(0x2545f4914f6cdd1d) >> 56) as u8 % 100;

                let is_source = raw[3] == 0x00;
                if is_source && r < LOSS_PCT {
                    dropped += 1;
                } else {
                    psock.send_to(&raw, rx_addr).await.unwrap();
                    forwarded += 1;
                }

                // Stop reporting after a burst — keeps the log readable.
                if (forwarded + dropped) % 64 == 0 {
                    eprintln!("  proxy: forwarded={forwarded} dropped={dropped}");
                }
            }
        })
    };

    let started = Instant::now();

    // ── Sender task ───────────────────────────────────────────────────────

    let blob_for_send = blob.clone();
    let send_task = tokio::spawn(async move {
        use delp::error::DelpError;
        use delp::transport::TransportError;

        for i in 0..n_symbols {
            let off = i * SYMBOL_SIZE;
            let chunk = Bytes::copy_from_slice(&blob_for_send[off..off + SYMBOL_SIZE]);

            // Retry on WindowFull — wait for feedback to evict ACK'd symbols
            // before pushing the next one.  Bounded sleep keeps the loop
            // making forward progress even on a slow ACK path.
            loop {
                let _ = sender.recv_feedback().await;
                match sender.send_source(chunk.clone()).await {
                    Ok(()) => break,
                    Err(TransportError::Codec(DelpError::WindowFull { .. })) => {
                        sleep(Duration::from_millis(2)).await;
                    }
                    Err(e) => panic!("send_source failed: {e}"),
                }
            }
        }
        // Flush — let in-flight feedback packets reach the encoder.
        sleep(Duration::from_millis(200)).await;
        let _ = sender.recv_feedback().await;
    });

    // ── Receiver task ─────────────────────────────────────────────────────

    let recv_task = tokio::spawn(async move {
        let mut reconstructed = vec![0u8; BLOB_BYTES];
        let mut received = 0usize;
        while received < n_symbols {
            let (id, data) = tokio::time::timeout(Duration::from_secs(10), receiver.recv_source())
                .await
                .expect("timeout waiting for symbol")
                .expect("recv_source error");
            let off = id as usize * SYMBOL_SIZE;
            reconstructed[off..off + SYMBOL_SIZE].copy_from_slice(&data);
            received += 1;
        }
        reconstructed
    });

    // ── Drive both halves to completion ───────────────────────────────────

    let (send_res, recv_res) = tokio::join!(send_task, recv_task);
    send_res?;
    let reconstructed = recv_res?;
    proxy.abort();

    let elapsed = started.elapsed();

    // ── Verification ──────────────────────────────────────────────────────

    if reconstructed == blob {
        println!("\n✓ {BLOB_BYTES} bytes reconstructed byte-exact");
    } else {
        let first_diff = blob.iter().zip(&reconstructed).position(|(a, b)| a != b);
        return Err(format!(
            "blob mismatch: first byte differs at offset {:?}",
            first_diff
        )
        .into());
    }

    println!(
        "  elapsed: {:.3} s ({:.2} MB/s effective throughput)",
        elapsed.as_secs_f64(),
        BLOB_BYTES as f64 / elapsed.as_secs_f64() / 1_000_000.0
    );

    Ok(())
}
