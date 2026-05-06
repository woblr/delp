//! [`DelpSession`] — bidirectional FEC session over a shared UDP socket.
//!
//! Combines a [`FecSender`] and a [`FecReceiver`] that share a single
//! `Arc<UdpSocket>`.  Call [`DelpSession::split`] to get independent halves
//! that can be driven concurrently in separate `tokio::spawn` tasks.
//!
//! # Topology
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │  Node A                           Node B                │
//! │  DelpSession                      DelpSession           │
//! │  ├─ FecSender ──── UDP ─────────► FecReceiver           │
//! │  │    (encoder)                   (decoder + feedback)  │
//! │  └─ FecReceiver ◄──── UDP ─────── FecSender             │
//! │       (decoder)                   (encoder)             │
//! └─────────────────────────────────────────────────────────┘
//! ```

use std::net::SocketAddr;
use std::sync::Arc;

use tokio::net::UdpSocket;

use crate::codec::decoder::Decoder;
use crate::codec::encoder::Encoder;
use crate::config::{DecoderConfig, EncoderConfig};
use crate::policy::{CongestionControl, FecRateController, FeedbackPolicy, WindowPolicy};

use super::{FecReceiver, FecSender, TransportResult};

// ── DelpSession ───────────────────────────────────────────────────────────

/// A full-duplex Delp session: sender + receiver over one socket.
pub struct DelpSession<W, C, F, P>
where
    W: WindowPolicy,
    C: CongestionControl,
    F: FecRateController,
    P: FeedbackPolicy,
{
    sender: FecSender<W, C, F>,
    receiver: FecReceiver<P>,
}

impl<W, C, F, P> DelpSession<W, C, F, P>
where
    W: WindowPolicy,
    C: CongestionControl,
    F: FecRateController,
    P: FeedbackPolicy,
{
    // ── Construction ─────────────────────────────────────────────────────

    /// Build a session from pre-constructed encoder and decoder.
    ///
    /// The `socket` is wrapped in an `Arc` and shared between sender and
    /// receiver so both halves can use the same OS socket handle without
    /// unsafe aliasing.
    pub fn new(
        encoder: Encoder<W, C, F>,
        decoder: Decoder<P>,
        socket: UdpSocket,
        remote: SocketAddr,
    ) -> Self {
        let socket = Arc::new(socket);
        Self {
            sender: FecSender::new(encoder, Arc::clone(&socket), remote),
            receiver: FecReceiver::new(decoder, socket, remote),
        }
    }

    // ── Split ─────────────────────────────────────────────────────────────

    /// Consume the session and return independent sender and receiver halves.
    ///
    /// Because both halves share an `Arc<UdpSocket>`, they can safely be
    /// moved to separate `tokio::spawn` tasks:
    ///
    /// ```rust,ignore
    /// let (mut tx, mut rx) = session.split();
    /// tokio::spawn(async move { tx.send_source(data).await.unwrap() });
    /// tokio::spawn(async move { rx.recv_source().await.unwrap()     });
    /// ```
    pub fn split(self) -> (FecSender<W, C, F>, FecReceiver<P>) {
        (self.sender, self.receiver)
    }

    // ── Convenience methods ───────────────────────────────────────────────

    pub fn sender(&self) -> &FecSender<W, C, F> {
        &self.sender
    }
    pub fn sender_mut(&mut self) -> &mut FecSender<W, C, F> {
        &mut self.sender
    }
    pub fn receiver(&self) -> &FecReceiver<P> {
        &self.receiver
    }
    pub fn receiver_mut(&mut self) -> &mut FecReceiver<P> {
        &mut self.receiver
    }
}

// ── Default session builder ───────────────────────────────────────────────

use crate::config::{BackpressureMode, Field, MatrixStrategy};
use crate::policy::defaults::{
    AnyAckPolicy, ConstantFecRate, ConstantFeedbackPolicy, NoCongestionControl,
};
use crate::{DefaultDecoder, DefaultEncoder};

/// Builder for the most common session configuration.
///
/// Uses default policies (AnyAck, no congestion control, constant FEC rate,
/// periodic feedback) and binds a new UDP socket for you.
///
/// ```rust,ignore
/// let session = SessionBuilder::new()
///     .symbol_size(1024)
///     .window_capacity(32)
///     .fec_rate(1, 2)        // 50 % redundancy
///     .bind("0.0.0.0:5000")
///     .remote("192.168.1.2:5001")
///     .build()
///     .await?;
/// ```
pub struct SessionBuilder {
    symbol_size: usize,
    window_capacity: usize,
    fec_numer: usize,
    fec_denom: usize,
    field: Field,
    strategy: MatrixStrategy,
    backpressure: BackpressureMode,
    feedback_every: u32,
}

impl Default for SessionBuilder {
    fn default() -> Self {
        Self {
            symbol_size: 1024,
            window_capacity: 32,
            fec_numer: 1,
            fec_denom: 4,
            field: Field::Gf2_8,
            strategy: MatrixStrategy::Vandermonde,
            backpressure: BackpressureMode::Reject,
            feedback_every: 8,
        }
    }
}

impl SessionBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn symbol_size(mut self, v: usize) -> Self {
        self.symbol_size = v;
        self
    }
    pub fn window_capacity(mut self, v: usize) -> Self {
        self.window_capacity = v;
        self
    }
    pub fn fec_rate(mut self, numer: usize, denom: usize) -> Self {
        self.fec_numer = numer;
        self.fec_denom = denom;
        self
    }
    pub fn field(mut self, f: Field) -> Self {
        self.field = f;
        self
    }
    pub fn strategy(mut self, s: MatrixStrategy) -> Self {
        self.strategy = s;
        self
    }
    pub fn feedback_every(mut self, n: u32) -> Self {
        self.feedback_every = n;
        self
    }

    /// Bind, connect and return a ready-to-use default session.
    pub async fn build(
        self,
        bind_addr: SocketAddr,
        remote: SocketAddr,
    ) -> TransportResult<
        DelpSession<AnyAckPolicy, NoCongestionControl, ConstantFecRate, ConstantFeedbackPolicy>,
    > {
        let enc_cfg = EncoderConfig::builder(self.symbol_size)
            .window_capacity(self.window_capacity)
            .fec_rate(self.fec_numer, self.fec_denom)
            .field(self.field)
            .matrix_strategy(self.strategy)
            .backpressure(self.backpressure)
            .build()?;

        let dec_cfg = DecoderConfig::builder(self.symbol_size)
            .field(self.field)
            .build()?;

        let encoder = DefaultEncoder::with_defaults(enc_cfg);
        let decoder = DefaultDecoder::with_defaults(dec_cfg);

        let socket = UdpSocket::bind(bind_addr).await?;

        Ok(DelpSession::new(encoder, decoder, socket, remote))
    }
}
