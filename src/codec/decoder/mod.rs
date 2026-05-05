pub mod buffer;
pub mod matrix;

use bytes::Bytes;
use std::collections::BTreeMap;
use tracing::{debug, trace};

use crate::config::DecoderConfig;
use crate::error::{Result, DelpError};
use crate::policy::{FeedbackPolicy, DecoderState, SourceSymbolId};
use crate::wire::{
    feedback::FeedbackPacket,
    source::SourcePacket,
    coded::CodedPacket,
};
use buffer::SymbolBuffer;
use matrix::DecodingMatrix;

// ── Decoder events ────────────────────────────────────────────────────────

/// Events emitted by the decoder after each packet is processed.
///
/// The caller must handle `SourceReady` in delivery order and should
/// transmit `SendFeedback` packets back to the encoder promptly.
#[derive(Debug, Clone)]
pub enum DecoderEvent {
    /// A source symbol is ready for delivery.
    ///
    /// IDs are emitted in strictly ascending order.
    SourceReady { id: SourceSymbolId, data: Bytes },
    /// The decoder requests that the caller send this feedback packet
    /// to the encoder (usually via the reverse path of the same transport).
    SendFeedback(Vec<u8>),
    /// A gap has opened that cannot be recovered — the encoder's window
    /// has advanced past these symbols.  The application must decide how
    /// to handle missing data (e.g. request retransmission at a higher level).
    UnrecoverableGap { first_missing: SourceSymbolId, count: u32 },
}

// ── Loss tracker ──────────────────────────────────────────────────────────

#[derive(Debug, Default)]
struct LossTracker {
    received:  u64,
    expected:  u64,
    last_seen: u32,
}

impl LossTracker {
    fn on_source(&mut self, id: u32) {
        self.received  += 1;
        let gap         = id.wrapping_sub(self.last_seen).min(1000) as u64;
        self.expected  += gap;
        self.last_seen  = id;
    }

    fn on_coded(&mut self) {
        self.received += 1;
        self.expected += 1;
    }

    fn loss_rate(&self) -> f64 {
        if self.expected == 0 { return 0.0; }
        let lost = self.expected.saturating_sub(self.received);
        (lost as f64 / self.expected as f64).clamp(0.0, 1.0)
    }
}

// ── Decoder ───────────────────────────────────────────────────────────────

/// RFC 9407 Delp decoder state machine.
///
/// Generic over [`FeedbackPolicy`] which controls how often feedback
/// (window update) packets are sent back to the encoder.
///
/// The decoder is a pure state machine — no I/O, no threads, no async.
/// Feed packets in (possibly out of order, possibly with gaps) and
/// collect [`DecoderEvent`]s.
pub struct Decoder<P: FeedbackPolicy> {
    config:          DecoderConfig,
    buffer:          SymbolBuffer,
    matrix:          DecodingMatrix,
    /// Known source symbols (received directly + recovered).
    known:           BTreeMap<SourceSymbolId, Bytes>,
    /// Lowest source ID the encoder may still have in its window.
    encoder_win_min: SourceSymbolId,
    loss:            LossTracker,
    feedback_policy: P,
    packets_received: u64,
}

impl<P: FeedbackPolicy> Decoder<P> {
    // ── Construction ─────────────────────────────────────────────────────

    pub fn new(config: DecoderConfig, feedback_policy: P) -> Self {
        let sym_size  = config.symbol_size;
        let field     = config.field;
        let max_rows  = config.max_matrix_rows;
        Self {
            buffer:          SymbolBuffer::new(0),
            matrix:          DecodingMatrix::new(field, sym_size, max_rows),
            known:           BTreeMap::new(),
            encoder_win_min: 0,
            loss:            LossTracker::default(),
            feedback_policy,
            packets_received: 0,
            config,
        }
    }

    // ── Public API ────────────────────────────────────────────────────────

    /// Process a received source packet.
    pub fn handle_source(&mut self, pkt: &SourcePacket<'_>) -> Result<Vec<DecoderEvent>> {
        if pkt.payload.len() != self.config.symbol_size {
            return Err(DelpError::SymbolSizeMismatch {
                expected: self.config.symbol_size,
                actual:   pkt.payload.len(),
            });
        }

        let id = pkt.source_symbol_id;
        trace!(id, "received source packet");
        self.loss.on_source(id);
        self.packets_received += 1;

        let data = Bytes::copy_from_slice(pkt.payload);

        // Store in known set and buffer
        self.known.entry(id).or_insert_with(|| data.clone());
        self.buffer.insert(id, data.clone());

        // Cascade into matrix
        let mut events = Vec::new();
        let recovered = self.matrix.add_known_source(id, data, &self.known)?;
        for (rid, rdata) in recovered {
            self.absorb_recovered(rid, rdata, &mut events);
        }

        // Drain deliverable
        self.drain_to_events(&mut events);
        self.maybe_send_feedback(&mut events);
        self.check_gap(&mut events);

        Ok(events)
    }

    /// Process a received coded packet.
    pub fn handle_coded(&mut self, pkt: &CodedPacket<'_>) -> Result<Vec<DecoderEvent>> {
        if pkt.payload.len() != self.config.symbol_size {
            return Err(DelpError::SymbolSizeMismatch {
                expected: self.config.symbol_size,
                actual:   pkt.payload.len(),
            });
        }

        trace!(coded_id = pkt.coded_symbol_id, "received coded packet");
        self.loss.on_coded();
        self.packets_received += 1;

        // Update encoder window min from the coding vector
        if let Some(&first_src) = pkt.ev.source_ids.first() {
            if first_src > self.encoder_win_min {
                self.encoder_win_min = first_src;
            }
        }

        let mut events = Vec::new();
        let recovered  = self.matrix.add_coded(&pkt.ev, pkt.payload, &self.known)?;
        for (rid, rdata) in recovered {
            self.absorb_recovered(rid, rdata, &mut events);
        }

        self.drain_to_events(&mut events);
        self.maybe_send_feedback(&mut events);
        self.check_gap(&mut events);

        Ok(events)
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    pub fn next_delivery_id(&self) -> SourceSymbolId {
        self.buffer.next_delivery_id
    }

    pub fn loss_rate(&self) -> f64 { self.loss.loss_rate() }
    pub fn packets_received(&self) -> u64 { self.packets_received }

    // ── Internals ────────────────────────────────────────────────────────

    fn absorb_recovered(
        &mut self,
        id:     SourceSymbolId,
        data:   Bytes,
        _events: &mut Vec<DecoderEvent>,
    ) {
        debug!(id, "recovered source symbol");
        self.known.entry(id).or_insert_with(|| data.clone());
        self.buffer.insert(id, data);
    }

    fn drain_to_events(&mut self, events: &mut Vec<DecoderEvent>) {
        for (id, data) in self.buffer.drain_deliverable() {
            events.push(DecoderEvent::SourceReady { id, data });
        }
    }

    fn maybe_send_feedback(&mut self, events: &mut Vec<DecoderEvent>) {
        let state = self.make_state();
        if !self.feedback_policy.should_send_feedback(state) { return; }

        let next  = self.buffer.next_delivery_id;
        let acked: Vec<u32> = self.buffer.ids_from(next).take(256).collect();
        let nb_missing = self.count_missing();

        let pkt = FeedbackPacket::build(
            next,
            nb_missing,
            self.matrix.row_count() as u32,
            self.loss.loss_rate(),
            &acked,
        );
        events.push(DecoderEvent::SendFeedback(pkt.serialise()));
    }

    fn check_gap(&mut self, events: &mut Vec<DecoderEvent>) {
        let next = self.buffer.next_delivery_id;
        if self.encoder_win_min > next {
            // Symbols [next .. encoder_win_min) are unrecoverable
            let count = self.encoder_win_min - next;
            events.push(DecoderEvent::UnrecoverableGap {
                first_missing: next,
                count,
            });
            // Advance delivery pointer past the gap
            self.buffer.prune_below(self.encoder_win_min);
        }
    }

    fn count_missing(&self) -> u32 {
        let next = self.buffer.next_delivery_id;
        let high = self.buffer.highest_id().unwrap_or(next);
        if high < next { return 0; }
        let span       = (high - next + 1) as u32;
        let buffered   = self.buffer.ids_from(next).count() as u32;
        span.saturating_sub(buffered)
    }

    fn make_state(&self) -> DecoderState {
        DecoderState {
            next_delivery_id:  self.buffer.next_delivery_id,
            packets_received:  self.packets_received,
            loss_rate:         self.loss.loss_rate(),
            nb_missing_src:    self.count_missing(),
            nb_not_used_coded: self.matrix.row_count() as u32,
        }
    }
}

// ── Convenience type alias ────────────────────────────────────────────────

use crate::policy::defaults::ConstantFeedbackPolicy;

/// Ready-to-use decoder with sensible defaults.
pub type DefaultDecoder = Decoder<ConstantFeedbackPolicy>;

impl DefaultDecoder {
    pub fn with_defaults(config: DecoderConfig) -> Self {
        Decoder::new(config, ConstantFeedbackPolicy::new(16))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use crate::config::{EncoderConfig, DecoderConfig, Field};
    use crate::codec::encoder::{DefaultEncoder, EncoderOutput};
    use crate::wire::{source::SourcePacket, coded::CodedPacket};

    fn run_round_trip(
        sym_size: usize,
        n_symbols: usize,
        field: Field,
        fec_numer: usize,
        fec_denom: usize,
        drop_indices: &[usize],
    ) -> Vec<(SourceSymbolId, Bytes)> {
        let enc_cfg = EncoderConfig::builder(sym_size)
            .field(field)
            .window_capacity(64)
            .fec_rate(fec_numer, fec_denom)
            .build()
            .unwrap();
        let dec_cfg = DecoderConfig::builder(sym_size)
            .field(field)
            .feedback_every(1000) // suppress feedback in tests
            .build()
            .unwrap();

        let mut enc = DefaultEncoder::with_defaults(enc_cfg);
        let mut dec = DefaultDecoder::with_defaults(dec_cfg);

        let payloads: Vec<Vec<u8>> = (0..n_symbols)
            .map(|i| vec![(i & 0xFF) as u8; sym_size])
            .collect();

        let mut all_pkts: Vec<Vec<u8>> = Vec::new();
        for payload in &payloads {
            let out = enc.submit_source(Bytes::copy_from_slice(payload)).unwrap();
            for item in out {
                match item {
                    EncoderOutput::Source(b) | EncoderOutput::Coded(b) => all_pkts.push(b),
                }
            }
        }

        let mut delivered: Vec<(SourceSymbolId, Bytes)> = Vec::new();
        for (i, pkt) in all_pkts.iter().enumerate() {
            if drop_indices.contains(&i) { continue; }
            // Determine type from header byte 3
            let pkt_type = pkt[3];
            let events = if pkt_type == 0x00 {
                let sp = SourcePacket::parse(pkt).unwrap();
                dec.handle_source(&sp).unwrap()
            } else {
                let cp = CodedPacket::parse(pkt).unwrap();
                dec.handle_coded(&cp).unwrap()
            };
            for ev in events {
                if let DecoderEvent::SourceReady { id, data } = ev {
                    delivered.push((id, data));
                }
            }
        }
        delivered.sort_by_key(|(id, _)| *id);
        delivered
    }

    #[test]
    fn no_loss_full_delivery() {
        let delivered = run_round_trip(64, 8, Field::Gf2_8, 0, 1, &[]);
        assert_eq!(delivered.len(), 8);
        for (i, (id, data)) in delivered.iter().enumerate() {
            assert_eq!(*id, i as u32);
            assert_eq!(data[0], (i & 0xFF) as u8);
        }
    }

    #[test]
    fn single_erasure_recovered_with_fec() {
        // Drop the 3rd source packet (index 2*2=4 with 1:1 FEC→ alternate src/coded)
        // With 1:1 FEC we have pkts: [src0, cod0, src1, cod1, src2, cod2, ...]
        // Drop src1 (index 2) — cod1 recovers it
        let delivered = run_round_trip(64, 4, Field::Gf2_8, 1, 1, &[2]);
        // All 4 source symbols should be delivered
        assert_eq!(delivered.len(), 4, "single erasure should be recovered");
    }

    #[test]
    fn gf2_4_no_loss() {
        let delivered = run_round_trip(64, 6, Field::Gf2_4, 0, 1, &[]);
        assert_eq!(delivered.len(), 6);
    }

    #[test]
    fn source_size_mismatch_rejected() {
        let cfg = DecoderConfig::builder(64).build().unwrap();
        let mut dec = DefaultDecoder::with_defaults(cfg);
        // Craft a source packet with wrong payload size
        let raw = SourcePacket::serialise(0, &vec![0u8; 32], &[], None);
        let sp  = SourcePacket::parse(&raw).unwrap();
        let err = dec.handle_source(&sp);
        assert!(matches!(err, Err(DelpError::SymbolSizeMismatch { .. })));
    }
}