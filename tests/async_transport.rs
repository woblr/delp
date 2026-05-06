//! Integration tests for the async UDP transport layer.
//!
//! These tests use real loopback UDP sockets to verify the full pipeline:
//! encode → transmit → receive → decode → feedback → window update.
//!
//! Run with:
//!   cargo test --features async --test async_transport

#[cfg(feature = "async")]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use bytes::Bytes;
    use tokio::net::UdpSocket;
    use tokio::time::timeout;

    use delp::codec::decoder::Decoder;
    use delp::codec::encoder::Encoder;
    use delp::config::{DecoderConfig, EncoderConfig, MatrixStrategy};
    use delp::policy::defaults::{
        AnyAckPolicy, ConstantFecRate, ConstantFeedbackPolicy, ImmediateFeedbackPolicy,
        NoCongestionControl,
    };
    use delp::transport::{FecReceiver, FecSender};

    // ── helpers ──────────────────────────────────────────────────────────

    async fn loopback_pair() -> (
        Arc<UdpSocket>,
        Arc<UdpSocket>,
        std::net::SocketAddr,
        std::net::SocketAddr,
    ) {
        let tx_sock = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let rx_sock = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let tx_addr = tx_sock.local_addr().unwrap();
        let rx_addr = rx_sock.local_addr().unwrap();
        (Arc::new(tx_sock), Arc::new(rx_sock), tx_addr, rx_addr)
    }

    /// Build an encoder with an *explicit* FEC rate (DefaultEncoder::with_defaults
    /// ignores config.fec_rate and always uses 1:4).
    fn make_encoder(
        sym_size: usize,
        window: usize,
        fec_numer: usize,
        fec_denom: usize,
    ) -> Encoder<AnyAckPolicy, NoCongestionControl, ConstantFecRate> {
        let cfg = EncoderConfig::builder(sym_size)
            .window_capacity(window)
            .matrix_strategy(MatrixStrategy::Vandermonde)
            .build()
            .unwrap();
        Encoder::new(
            cfg,
            AnyAckPolicy,
            NoCongestionControl,
            ConstantFecRate::new(fec_numer, fec_denom),
        )
    }

    /// Decoder with periodic feedback (default period = 4 packets).
    fn make_decoder(sym_size: usize) -> Decoder<ConstantFeedbackPolicy> {
        let cfg = DecoderConfig::builder(sym_size).build().unwrap();
        Decoder::new(cfg, ConstantFeedbackPolicy::new(4))
    }

    /// Decoder that emits feedback after every single packet — needed when
    /// the test sends fewer packets than the default feedback period.
    fn make_decoder_immediate(sym_size: usize) -> Decoder<ImmediateFeedbackPolicy> {
        let cfg = DecoderConfig::builder(sym_size).build().unwrap();
        Decoder::new(cfg, ImmediateFeedbackPolicy)
    }

    fn payload(sym_size: usize, seed: u8) -> Bytes {
        Bytes::from(vec![seed; sym_size])
    }

    // ── tests ─────────────────────────────────────────────────────────────

    /// No packet loss — all symbols delivered in order.
    #[tokio::test]
    async fn round_trip_no_loss() {
        const SYM: usize = 256;
        const N: usize = 8;

        let (tx_sock, rx_sock, tx_addr, rx_addr) = loopback_pair().await;

        let encoder = make_encoder(SYM, N, 1, 2); // 50% FEC
        let decoder = make_decoder(SYM);

        let mut sender = FecSender::new(encoder, tx_sock, rx_addr);
        let mut receiver = FecReceiver::new(decoder, rx_sock, tx_addr);

        // Send N symbols.
        for i in 0..N as u8 {
            sender.send_source(payload(SYM, i)).await.unwrap();
        }

        // Receive all N symbols with a timeout.
        for i in 0..N as u8 {
            let (id, data) = timeout(Duration::from_secs(2), receiver.recv_source())
                .await
                .expect("timed out waiting for symbol")
                .unwrap();
            assert_eq!(id, i as u32);
            assert_eq!(data.as_ref(), &vec![i; SYM]);
        }
    }

    /// Source packets are all dropped; FEC coded packets carry recovery.
    #[tokio::test]
    async fn round_trip_with_source_packet_loss() {
        const SYM: usize = 512;
        const N: usize = 6;
        // 1:1 FEC → N source + N coded = 2N datagrams hit the proxy.
        const TOTAL: usize = N * 2;

        let tx_sock = Arc::new(UdpSocket::bind("127.0.0.1:0").await.unwrap());
        let rx_sock = Arc::new(UdpSocket::bind("127.0.0.1:0").await.unwrap());
        let prx_sock = Arc::new(UdpSocket::bind("127.0.0.1:0").await.unwrap());

        let tx_addr = tx_sock.local_addr().unwrap();
        let rx_addr = rx_sock.local_addr().unwrap();
        let prx_addr = prx_sock.local_addr().unwrap();

        let encoder = make_encoder(SYM, N, 1, 1);
        let decoder = make_decoder(SYM);

        let mut sender = FecSender::new(encoder, tx_sock, prx_addr);
        let mut receiver = FecReceiver::new(decoder, rx_sock, tx_addr);

        // Async proxy: receive TOTAL packets, drop source, forward coded.
        let proxy_task = tokio::spawn(async move {
            let mut buf = vec![0u8; 65536];
            let mut fwd = 0usize;
            for _ in 0..TOTAL {
                let (n, _) = prx_sock.recv_from(&mut buf).await.unwrap();
                let raw = buf[..n].to_vec();
                if raw.len() >= 4 && raw[3] == 0x01 {
                    prx_sock.send_to(&raw, rx_addr).await.unwrap();
                    fwd += 1;
                }
                // source packets (0x00) silently dropped
            }
            fwd
        });

        // Send N symbols — 1:1 FEC emits 1 source + 1 coded per symbol.
        for i in 0..N as u8 {
            sender.send_source(payload(SYM, i)).await.unwrap();
        }

        let fwd = timeout(Duration::from_secs(5), proxy_task)
            .await
            .expect("proxy task timed out")
            .unwrap();
        assert_eq!(fwd, N, "expected {N} coded pkts forwarded, got {fwd}");

        // All N symbols must be recoverable from coded packets alone.
        for _ in 0..N {
            timeout(Duration::from_secs(5), receiver.recv_source())
                .await
                .expect("timed out — FEC recovery failed")
                .unwrap();
        }
    }

    /// Verify feedback flows from receiver back to sender.
    #[tokio::test]
    async fn feedback_reaches_sender() {
        const SYM: usize = 64;
        const N: usize = 4;

        let (tx_sock, rx_sock, tx_addr, rx_addr) = loopback_pair().await;

        let encoder = make_encoder(SYM, N, 1, 1);
        // ImmediateFeedbackPolicy: emit feedback after every packet so the
        // sender receives at least one within the small N=4 test window.
        let decoder = make_decoder_immediate(SYM);

        let mut sender = FecSender::new(encoder, tx_sock, rx_addr);
        let mut receiver = FecReceiver::new(decoder, rx_sock, tx_addr);

        for i in 0..N as u8 {
            sender.send_source(payload(SYM, i)).await.unwrap();
        }

        // Drain all received symbols (this also sends feedback).
        for _ in 0..N {
            timeout(Duration::from_secs(2), receiver.recv_source())
                .await
                .unwrap()
                .unwrap();
        }

        // Sender should receive at least one feedback packet.
        // Use wait_feedback (blocking recv) so we don't miss it.
        timeout(Duration::from_secs(2), sender.wait_feedback())
            .await
            .expect("no feedback received by sender")
            .unwrap();
    }

    /// Stream trait: collect symbols via `futures_util::StreamExt::next`.
    #[tokio::test]
    async fn stream_trait_delivers_symbols() {
        use futures_util::StreamExt;

        const SYM: usize = 128;
        const N: usize = 4;

        let (tx_sock, rx_sock, tx_addr, rx_addr) = loopback_pair().await;

        let encoder = make_encoder(SYM, N, 0, 1); // no FEC, source only
        let decoder = make_decoder(SYM);

        let mut sender = FecSender::new(encoder, tx_sock, rx_addr);
        let mut receiver = FecReceiver::new(decoder, rx_sock, tx_addr);

        for i in 0..N as u8 {
            sender.send_source(payload(SYM, i)).await.unwrap();
        }

        let mut received = 0usize;
        while received < N {
            let sym = timeout(Duration::from_secs(2), receiver.next())
                .await
                .expect("stream timed out")
                .expect("stream ended unexpectedly")
                .unwrap();
            assert_eq!(sym.0 as usize, received);
            received += 1;
        }
        assert_eq!(received, N);
    }

    /// DelpSession builder: verify construction succeeds.
    #[tokio::test]
    async fn session_builder_constructs() {
        use delp::transport::session::SessionBuilder;

        let sock_a = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let sock_b = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let addr_a = sock_a.local_addr().unwrap();
        let addr_b = sock_b.local_addr().unwrap();
        drop(sock_a);
        drop(sock_b);

        let session = SessionBuilder::new()
            .symbol_size(256)
            .window_capacity(8)
            .fec_rate(1, 2)
            .build(addr_a, addr_b)
            .await
            .unwrap();

        let (_tx, _rx) = session.split();
    }
}
