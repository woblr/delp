//! Minimal end-to-end FEC example over loopback UDP.
//!
//! Runs a sender and receiver concurrently in two tokio tasks, separated
//! by a simulated 30 % packet loss channel.  All source symbols are
//! recovered via FEC coded packets.
//!
//! Usage:
//!   cargo run --example udp_fec --features async

#[cfg(not(feature = "async"))]
fn main() {
    eprintln!("This example requires --features async");
    std::process::exit(1);
}

#[cfg(feature = "async")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::Arc;
    use std::time::Duration;

    use bytes::Bytes;
    use tokio::net::UdpSocket;
    use tokio::time::sleep;

    use delp::config::{DecoderConfig, EncoderConfig, MatrixStrategy};
    use delp::transport::{FecReceiver, FecSender};
    use delp::{DefaultDecoder, DefaultEncoder};

    // ── Configuration ─────────────────────────────────────────────────────

    const SYMBOL_SIZE: usize = 1024; // bytes per source symbol
    const WINDOW: usize = 16; // symbols in the encoding window
    const N_SYMBOLS: usize = 12; // total source symbols to send
    const LOSS_PCT: u8 = 30; // simulated packet loss %

    // ── Sockets ───────────────────────────────────────────────────────────

    // sender_sock ──► [lossy proxy] ──► receiver_sock
    //     ◄────────────── feedback ───────────────────

    let sender_sock = Arc::new(UdpSocket::bind("127.0.0.1:0").await?);
    let proxy_sock = Arc::new(UdpSocket::bind("127.0.0.1:0").await?);
    let receiver_sock = Arc::new(UdpSocket::bind("127.0.0.1:0").await?);

    let sender_addr = sender_sock.local_addr()?;
    let proxy_addr = proxy_sock.local_addr()?;
    let receiver_addr = receiver_sock.local_addr()?;

    println!("Sender:   {sender_addr}");
    println!("Proxy:    {proxy_addr}  ({LOSS_PCT}% loss)");
    println!("Receiver: {receiver_addr}");
    println!("Symbols:  {N_SYMBOLS} × {SYMBOL_SIZE} bytes, window={WINDOW}, 1:1 FEC");
    println!();

    // ── Encoder / Decoder ─────────────────────────────────────────────────

    let enc_cfg = EncoderConfig::builder(SYMBOL_SIZE)
        .window_capacity(WINDOW)
        .fec_rate(1, 1) // 100% FEC: 1 coded per source
        .matrix_strategy(MatrixStrategy::Vandermonde)
        .build()?;

    let dec_cfg = DecoderConfig::builder(SYMBOL_SIZE).build()?;

    let encoder = DefaultEncoder::with_defaults(enc_cfg);
    let decoder = DefaultDecoder::with_defaults(dec_cfg);

    let mut sender = FecSender::new(encoder, Arc::clone(&sender_sock), proxy_addr);
    let mut receiver = FecReceiver::new(decoder, Arc::clone(&receiver_sock), sender_addr);

    // ── Lossy proxy task ──────────────────────────────────────────────────
    //
    // Forwards packets from sender to receiver, randomly dropping LOSS_PCT%
    // of them.  Feedback packets from receiver are forwarded back to sender
    // without loss.

    let proxy = {
        let psock = Arc::clone(&proxy_sock);
        let rx_addr = receiver_addr;
        let tx_addr = sender_addr;
        tokio::spawn(async move {
            let mut buf = vec![0u8; 65536];
            let mut total = 0u32;
            loop {
                let (n, _src) = psock.recv_from(&mut buf).await.unwrap();
                let raw = buf[..n].to_vec();
                total += 1;

                // Feedback: always forward back to sender.
                if raw.len() >= 4 && raw[3] == 0x02 {
                    psock.send_to(&raw, tx_addr).await.unwrap();
                    continue;
                }

                // Drop source packets at the configured loss rate; always
                // forward coded packets so the receiver can recover lost
                // sources via the FEC equations.
                let is_source = raw[3] == 0x00;
                let drop = is_source && (rand_u8() % 100) < LOSS_PCT;
                if drop {
                    println!("  proxy: dropped source pkt #{total}");
                } else {
                    psock.send_to(&raw, rx_addr).await.unwrap();
                }
            }
        })
    };

    // ── Sender task ───────────────────────────────────────────────────────

    let send_task = tokio::spawn(async move {
        for i in 0..N_SYMBOLS {
            let data = Bytes::from(
                format!("symbol-{i:04}")
                    .into_bytes()
                    .into_iter()
                    .cycle()
                    .take(SYMBOL_SIZE)
                    .collect::<Vec<_>>(),
            );
            sender.send_source(data).await.unwrap();
            println!("  sent source symbol {i}");
            sleep(Duration::from_millis(5)).await;
        }
        // Give the receiver time to drain and send feedback.
        sleep(Duration::from_millis(500)).await;
        let fb = sender.recv_feedback().await.unwrap();
        println!("\nSender received {fb} feedback packet(s)");
    });

    // ── Receiver task ─────────────────────────────────────────────────────

    let recv_task = tokio::spawn(async move {
        for expected_id in 0..N_SYMBOLS {
            let (id, data) = tokio::time::timeout(Duration::from_secs(5), receiver.recv_source())
                .await
                .expect("timeout — FEC recovery failed")
                .unwrap();

            let preview = String::from_utf8_lossy(&data[..12.min(data.len())]);
            println!("  recv  source symbol {id}  [{preview}...]");
            assert_eq!(id as usize, expected_id, "out-of-order delivery");
        }
        println!("\nAll {N_SYMBOLS} symbols delivered in order.");
    });

    // ── Run ───────────────────────────────────────────────────────────────

    let (sr, rr) = tokio::join!(send_task, recv_task);
    sr?;
    rr?;
    proxy.abort();

    println!("\nDone.  FEC successfully recovered all symbols despite {LOSS_PCT}% packet loss.");
    Ok(())
}

/// Minimal PRNG — avoids pulling in `rand` as a non-dev dependency.
///
/// Seeded from the system clock at first call so consecutive runs of the
/// example exercise different drop patterns.
#[cfg(feature = "async")]
fn rand_u8() -> u8 {
    use std::cell::Cell;
    use std::time::{SystemTime, UNIX_EPOCH};
    thread_local! {
        static STATE: Cell<u64> = const { Cell::new(0) };
    }
    STATE.with(|s| {
        let mut x = s.get();
        if x == 0 {
            x = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x853c49e6748fea9b);
            if x == 0 {
                x = 0x853c49e6748fea9b;
            }
        }
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        s.set(x);
        (x.wrapping_mul(0x2545f4914f6cdd1d) >> 56) as u8
    })
}
