//! [`FecReceiver`] — receives UDP packets, decodes FEC, delivers symbols in order.
//!
//! Each datagram is dispatched through the [`crate::codec::decoder::Decoder`]
//! state machine.  Recovered source symbols are queued internally and exposed
//! via both a blocking [`FecReceiver::recv_source`] method and a
//! [`futures_core::Stream`] impl for use in async pipelines.
//!
//! Feedback packets are sent back to the encoder automatically whenever the
//! decoder's [`crate::policy::FeedbackPolicy`] requests it.

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use bytes::Bytes;
use tokio::net::UdpSocket;

use crate::codec::decoder::{Decoder, DecoderEvent};
use crate::policy::{FeedbackPolicy, SourceSymbolId};
use crate::wire::coded::CodedPacket;
use crate::wire::common::{CommonHeader, PacketType};
use crate::wire::source::SourcePacket;

use super::{TransportError, TransportResult};

// ── FecReceiver ───────────────────────────────────────────────────────────

/// Async FEC receiver wrapping a [`Decoder`] over a UDP socket.
///
/// # Generic parameters
///
/// - `P` — [`FeedbackPolicy`]: controls when window-update packets are sent
///   back to the encoder.
pub struct FecReceiver<P: FeedbackPolicy> {
    decoder: Decoder<P>,
    socket: Arc<UdpSocket>,
    encoder_addr: SocketAddr,
    /// Decoded symbols waiting to be consumed by the caller.
    ready: VecDeque<(SourceSymbolId, Bytes)>,
    /// Reusable datagram receive buffer (max UDP payload = 65 507 bytes).
    rx_buf: Vec<u8>,
}

impl<P: FeedbackPolicy> FecReceiver<P> {
    // ── Construction ─────────────────────────────────────────────────────

    /// Create a new receiver.
    ///
    /// `socket` must already be bound.  `encoder_addr` is the address of
    /// the remote sender; feedback packets are sent there.
    pub fn new(decoder: Decoder<P>, socket: Arc<UdpSocket>, encoder_addr: SocketAddr) -> Self {
        Self {
            decoder,
            socket,
            encoder_addr,
            ready: VecDeque::new(),
            rx_buf: vec![0u8; 65536],
        }
    }

    // ── Receiving ─────────────────────────────────────────────────────────

    /// Wait for and process one UDP datagram.
    ///
    /// May return zero or more recovered symbols in `ready` as a side effect.
    /// Feedback packets are sent back to `encoder_addr` automatically.
    ///
    /// Returns `true` if at least one source symbol became ready.
    async fn process_one(&mut self) -> TransportResult<bool> {
        let (n, _src) = self.socket.recv_from(&mut self.rx_buf).await?;
        let raw = &self.rx_buf[..n];

        // Determine packet type from common header.
        let hdr = match CommonHeader::parse(raw) {
            Ok(h) => h,
            Err(_) => return Ok(false), // silently drop malformed datagrams
        };

        let pkt_type = match hdr.packet_type() {
            Ok(t) => t,
            Err(_) => return Ok(false),
        };

        let events = match pkt_type {
            PacketType::Source => match SourcePacket::parse(raw) {
                Ok(sp) => self.decoder.handle_source(&sp)?,
                Err(_) => return Ok(false),
            },
            PacketType::Coded => match CodedPacket::parse(raw) {
                Ok(cp) => self.decoder.handle_coded(&cp)?,
                Err(_) => return Ok(false),
            },
            PacketType::Feedback => {
                // Feedback arriving at the receiver is unexpected but harmless.
                return Ok(false);
            }
        };

        let mut became_ready = false;
        for ev in events {
            match ev {
                DecoderEvent::SourceReady { id, data } => {
                    self.ready.push_back((id, data));
                    became_ready = true;
                }
                DecoderEvent::SendFeedback(raw_fb) => {
                    // Best-effort — ignore send failures (fire and forget).
                    let _ = self.socket.send_to(&raw_fb, self.encoder_addr).await;
                }
                DecoderEvent::UnrecoverableGap { .. } => {
                    // Gap events are silently swallowed at the transport layer.
                    // Higher-level protocols should handle retransmission.
                }
            }
        }
        Ok(became_ready)
    }

    /// Receive the next delivered source symbol, waiting as long as necessary.
    ///
    /// Symbols are delivered in strictly ascending ID order.
    pub async fn recv_source(&mut self) -> TransportResult<(SourceSymbolId, Bytes)> {
        loop {
            if let Some(sym) = self.ready.pop_front() {
                return Ok(sym);
            }
            self.process_one().await?;
        }
    }

    /// Non-blocking: return a symbol if one is already decoded, without
    /// waiting for more network data.
    pub fn try_recv_source(&mut self) -> Option<(SourceSymbolId, Bytes)> {
        self.ready.pop_front()
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    pub fn decoder(&self) -> &Decoder<P> {
        &self.decoder
    }
    pub fn decoder_mut(&mut self) -> &mut Decoder<P> {
        &mut self.decoder
    }
    pub fn socket(&self) -> &Arc<UdpSocket> {
        &self.socket
    }
    pub fn encoder_addr(&self) -> SocketAddr {
        self.encoder_addr
    }

    /// How many decoded symbols are queued and waiting for the caller.
    pub fn ready_count(&self) -> usize {
        self.ready.len()
    }
}

// ── futures_core::Stream ──────────────────────────────────────────────────
//
// Allows FecReceiver to be used as a `Stream<Item = TransportResult<(u32, Bytes)>>`
// in async pipelines (`StreamExt::next`, `pin_mut!`, etc.).

impl<P: FeedbackPolicy + Unpin> futures_core::Stream for FecReceiver<P> {
    type Item = TransportResult<(SourceSymbolId, Bytes)>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // 1. Return any already-decoded symbol immediately.
        if let Some(sym) = self.ready.pop_front() {
            return Poll::Ready(Some(Ok(sym)));
        }

        // 2. Clone the socket Arc so the borrow checker sees it as
        //    independent of `self.rx_buf`.
        let sock = Arc::clone(&self.socket);

        // 3. Try a non-blocking recv into a temporary local buffer to avoid
        //    holding a borrow into `self.rx_buf` while calling `dispatch_datagram`.
        let mut tmp = vec![0u8; 65536];
        match sock.try_recv_from(&mut tmp) {
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                let waker = cx.waker().clone();
                tokio::spawn(async move {
                    let _ = sock.readable().await;
                    waker.wake();
                });
                Poll::Pending
            }
            Err(e) => Poll::Ready(Some(Err(TransportError::Io(e)))),
            Ok((n, _src)) => {
                let raw = tmp[..n].to_vec();
                let result = self.dispatch_datagram(&raw);
                match result {
                    Err(e) => Poll::Ready(Some(Err(e))),
                    Ok(Some(sym)) => Poll::Ready(Some(Ok(sym))),
                    Ok(None) => {
                        cx.waker().wake_by_ref();
                        Poll::Pending
                    }
                }
            }
        }
    }
}

impl<P: FeedbackPolicy> FecReceiver<P> {
    /// Synchronous datagram dispatch (no async feedback sending in Stream path).
    ///
    /// Feedback is queued and sent lazily; in the Stream path we fire-and-forget
    /// via `try_send_to` rather than `.await`-ing.
    fn dispatch_datagram(
        &mut self,
        raw: &[u8],
    ) -> TransportResult<Option<(SourceSymbolId, Bytes)>> {
        let hdr = match CommonHeader::parse(raw) {
            Ok(h) => h,
            Err(_) => return Ok(None),
        };
        let pkt_type = match hdr.packet_type() {
            Ok(t) => t,
            Err(_) => return Ok(None),
        };

        let events = match pkt_type {
            PacketType::Source => match SourcePacket::parse(raw) {
                Ok(sp) => self.decoder.handle_source(&sp)?,
                Err(_) => return Ok(None),
            },
            PacketType::Coded => match CodedPacket::parse(raw) {
                Ok(cp) => self.decoder.handle_coded(&cp)?,
                Err(_) => return Ok(None),
            },
            PacketType::Feedback => return Ok(None),
        };

        let encoder_addr = self.encoder_addr;
        for ev in events {
            match ev {
                DecoderEvent::SourceReady { id, data } => {
                    self.ready.push_back((id, data));
                }
                DecoderEvent::SendFeedback(fb) => {
                    // Best-effort non-blocking send.
                    let _ = self.socket.try_send_to(&fb, encoder_addr);
                }
                DecoderEvent::UnrecoverableGap { .. } => {}
            }
        }

        Ok(self.ready.pop_front())
    }
}
