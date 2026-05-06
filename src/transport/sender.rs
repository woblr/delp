//! [`FecSender`] — encodes source symbols and transmits them over UDP.
//!
//! Each call to [`FecSender::send_source`] encodes one source symbol,
//! dispatches the resulting source + coded packets to the remote address,
//! and returns only when all bytes are on the wire.
//!
//! Feedback packets arriving from the remote decoder are drained via
//! [`FecSender::recv_feedback`].  Call it periodically (e.g. in a `select!`
//! branch) to keep the encoder's window and rate controller up to date.

use std::net::SocketAddr;
use std::sync::Arc;

use bytes::Bytes;
use tokio::net::UdpSocket;

use crate::codec::encoder::{Encoder, EncoderOutput};
use crate::policy::{CongestionControl, FecRateController, ReceiverId, WindowPolicy};
use crate::wire::feedback::FeedbackPacket;

use super::{TransportError, TransportResult};

// ── FecSender ────────────────────────────────────────────────────────────

/// Async FEC sender wrapping an [`Encoder`] over a UDP socket.
///
/// # Generic parameters
///
/// - `W` — [`WindowPolicy`]: controls which symbols are evicted from the
///   encoding window after acknowledgement.
/// - `C` — [`CongestionControl`]: rate-limits transmission.
/// - `F` — [`FecRateController`]: decides how many coded packets to emit
///   per source packet.
pub struct FecSender<W, C, F>
where
    W: WindowPolicy,
    C: CongestionControl,
    F: FecRateController,
{
    encoder: Encoder<W, C, F>,
    socket: Arc<UdpSocket>,
    remote: SocketAddr,
    /// Stable receiver ID derived from the remote address.
    receiver_id: ReceiverId,
    /// Temporary receive buffer — reused across [`recv_feedback`] calls.
    rx_buf: Vec<u8>,
}

impl<W, C, F> FecSender<W, C, F>
where
    W: WindowPolicy,
    C: CongestionControl,
    F: FecRateController,
{
    // ── Construction ─────────────────────────────────────────────────────

    /// Create a new sender.
    ///
    /// `socket` must already be bound.  `remote` is the destination address
    /// of the decoder.
    pub fn new(encoder: Encoder<W, C, F>, socket: Arc<UdpSocket>, remote: SocketAddr) -> Self {
        let receiver_id = addr_to_receiver_id(remote);
        Self {
            encoder,
            socket,
            remote,
            receiver_id,
            rx_buf: vec![0u8; 65536],
        }
    }

    // ── Source transmission ───────────────────────────────────────────────

    /// Encode and send one source symbol.
    ///
    /// Emits one `Source` packet plus zero or more `Coded` packets as
    /// determined by the [`FecRateController`].  All packets are sent
    /// before this method returns.
    ///
    /// # Errors
    ///
    /// - [`TransportError::Codec`] if the symbol size does not match the
    ///   encoder configuration, or if the window is full and backpressure
    ///   policy is `Reject`.
    /// - [`TransportError::Io`] on socket send failure.
    pub async fn send_source(&mut self, data: Bytes) -> TransportResult<()> {
        let packets = self.encoder.submit_source(data)?;
        for pkt in packets {
            let raw = match pkt {
                EncoderOutput::Source(b) => b,
                EncoderOutput::Coded(b) => b,
            };
            self.socket.send_to(&raw, self.remote).await?;
        }
        Ok(())
    }

    /// Attempt to generate and send one additional coded packet.
    ///
    /// Returns `Ok(true)` if a coded packet was sent, `Ok(false)` if the
    /// window is empty, and `Err` on exhaustion or socket failure.
    pub async fn send_coded(&mut self) -> TransportResult<bool> {
        match self.encoder.generate_coded()? {
            None => Ok(false),
            Some(EncoderOutput::Coded(raw)) => {
                self.socket.send_to(&raw, self.remote).await?;
                Ok(true)
            }
            Some(EncoderOutput::Source(raw)) => {
                self.socket.send_to(&raw, self.remote).await?;
                Ok(true)
            }
        }
    }

    // ── Feedback reception ────────────────────────────────────────────────

    /// Non-blocking: drain all pending feedback packets from the socket.
    ///
    /// Call this inside a `tokio::select!` branch or after sending a batch
    /// of packets to keep the encoder's window and rate controller updated.
    ///
    /// Returns the number of feedback packets processed.
    pub async fn recv_feedback(&mut self) -> TransportResult<usize> {
        let mut count = 0;
        loop {
            match self.socket.try_recv_from(&mut self.rx_buf) {
                Ok((n, src)) => {
                    let raw = &self.rx_buf[..n];
                    if let Ok(fp) = FeedbackPacket::parse(raw) {
                        let rid = addr_to_receiver_id(src);
                        self.encoder.handle_feedback(rid, &fp);
                        count += 1;
                    }
                    // silently drop malformed feedback — lossy network is expected
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(e) => return Err(TransportError::Io(e)),
            }
        }
        Ok(count)
    }

    /// Blocking: wait for the next feedback packet and process it.
    pub async fn wait_feedback(&mut self) -> TransportResult<()> {
        let (n, src) = self.socket.recv_from(&mut self.rx_buf).await?;
        let raw = &self.rx_buf[..n];
        if let Ok(fp) = FeedbackPacket::parse(raw) {
            let rid = addr_to_receiver_id(src);
            self.encoder.handle_feedback(rid, &fp);
        }
        Ok(())
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    pub fn encoder(&self) -> &Encoder<W, C, F> {
        &self.encoder
    }
    pub fn encoder_mut(&mut self) -> &mut Encoder<W, C, F> {
        &mut self.encoder
    }
    pub fn socket(&self) -> &Arc<UdpSocket> {
        &self.socket
    }
    pub fn remote(&self) -> SocketAddr {
        self.remote
    }
    pub fn receiver_id(&self) -> ReceiverId {
        self.receiver_id
    }
}

// ── futures_sink::Sink<Bytes> ─────────────────────────────────────────────
//
// Allows FecSender to be used as a `Sink` in a futures pipeline.
// `start_send` encodes and queues packets; `poll_flush` drains the queue
// over the UDP socket.

use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Sink-capable sender wraps [`FecSender`] with a packet queue.
///
/// Use [`SinkSender::new`] and call [`futures_util::SinkExt::send`] to send
/// source symbols through a `Sink<Bytes>` interface.
pub struct SinkSender<W, C, F>
where
    W: WindowPolicy,
    C: CongestionControl,
    F: FecRateController,
{
    inner: FecSender<W, C, F>,
    pending: VecDeque<Vec<u8>>,
}

impl<W, C, F> SinkSender<W, C, F>
where
    W: WindowPolicy,
    C: CongestionControl,
    F: FecRateController,
{
    pub fn new(encoder: Encoder<W, C, F>, socket: Arc<UdpSocket>, remote: SocketAddr) -> Self {
        Self {
            inner: FecSender::new(encoder, socket, remote),
            pending: VecDeque::new(),
        }
    }

    pub fn into_inner(self) -> FecSender<W, C, F> {
        self.inner
    }
}

impl<W, C, F> futures_sink::Sink<Bytes> for SinkSender<W, C, F>
where
    W: WindowPolicy + Unpin,
    C: CongestionControl + Unpin,
    F: FecRateController + Unpin,
{
    type Error = TransportError;

    fn poll_ready(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        // Apply back-pressure: stop accepting if the queue is very large.
        // In practice, the window capacity provides the primary back-pressure.
        const MAX_PENDING: usize = 512;
        if self.pending.len() >= MAX_PENDING {
            Poll::Pending
        } else {
            Poll::Ready(Ok(()))
        }
    }

    fn start_send(mut self: Pin<&mut Self>, data: Bytes) -> Result<(), Self::Error> {
        let packets = self.inner.encoder.submit_source(data)?;
        for pkt in packets {
            let raw = match pkt {
                EncoderOutput::Source(b) => b,
                EncoderOutput::Coded(b) => b,
            };
            self.pending.push_back(raw);
        }
        Ok(())
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        let remote = self.inner.remote;
        while let Some(pkt) = self.pending.front() {
            match self.inner.socket.try_send_to(pkt, remote) {
                Ok(_) => {
                    self.pending.pop_front();
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    // Socket not ready — register waker and retry later.
                    let sock = Arc::clone(&self.inner.socket);
                    let waker = cx.waker().clone();
                    tokio::spawn(async move {
                        let _ = sock.writable().await;
                        waker.wake();
                    });
                    return Poll::Pending;
                }
                Err(e) => return Poll::Ready(Err(TransportError::Io(e))),
            }
        }
        Poll::Ready(Ok(()))
    }

    fn poll_close(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.poll_flush(cx)
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

/// Derive a stable `ReceiverId` from a socket address.
///
/// Simple multiplicative hash — not cryptographic, purely for in-process
/// demultiplexing of feedback packets from multiple receivers.
fn addr_to_receiver_id(addr: SocketAddr) -> ReceiverId {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    addr.hash(&mut h);
    h.finish()
}
