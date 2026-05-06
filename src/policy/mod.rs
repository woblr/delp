pub mod defaults;

pub use defaults::{
    AllAckPolicy, AnyAckPolicy, ConstantFecRate, ConstantFeedbackPolicy, NoCongestionControl,
};

use crate::wire::feedback::FeedbackPacket;

// ── Type aliases ──────────────────────────────────────────────────────────

pub type SourceSymbolId = u32;
pub type CodedSymbolId = u32;
pub type ReceiverId = u64;

// ── State snapshots (borrow-split helpers) ────────────────────────────────

/// Read-only view of encoder state passed to policy callbacks.
///
/// Using a separate struct prevents borrow conflicts when the encoder
/// simultaneously borrows its policy fields mutably.
#[derive(Debug, Clone, Copy)]
pub struct EncoderState<'a> {
    pub window_size: usize,
    pub window_capacity: usize,
    pub next_source_id: SourceSymbolId,
    pub next_coded_id: CodedSymbolId,
    /// Last measured packet loss rate per receiver (0.0..=1.0).
    pub loss_rate: f64,
    /// Source IDs currently in the encoding window.
    pub window_ids: &'a [SourceSymbolId],
}

/// Read-only view of decoder state passed to policy callbacks.
#[derive(Debug, Clone, Copy)]
pub struct DecoderState {
    pub next_delivery_id: SourceSymbolId,
    pub packets_received: u64,
    pub loss_rate: f64,
    pub nb_missing_src: u32,
    pub nb_not_used_coded: u32,
}

// ── Per-receiver ACK tracking (shared between encoder and window policy) ──

/// Tracks which source IDs each receiver has acknowledged.
#[derive(Debug, Default)]
pub struct ReceiverAckState {
    /// Map from receiver ID to highest contiguous ACK + SACK set.
    receivers: std::collections::HashMap<ReceiverId, ReceiverEntry>,
}

#[derive(Debug, Default)]
struct ReceiverEntry {
    /// Cumulative ACK: all IDs < this value are acknowledged.
    cumulative_ack: SourceSymbolId,
    /// Selective ACK: IDs ≥ cumulative_ack that are individually ACKed.
    sack: std::collections::BTreeSet<SourceSymbolId>,
}

impl ReceiverAckState {
    pub fn update(&mut self, receiver_id: ReceiverId, pkt: &FeedbackPacket) {
        let entry = self.receivers.entry(receiver_id).or_default();

        // Advance cumulative ACK from first_src_id
        if pkt.first_src_id > entry.cumulative_ack {
            entry.cumulative_ack = pkt.first_src_id;
            // Remove SACK entries now covered by cumulative ACK
            entry.sack.retain(|&id| id >= entry.cumulative_ack);
        }

        // Add SACK bits
        for id in pkt.acked_ids() {
            entry.sack.insert(id);
        }

        // Advance cumulative ACK through contiguous SACK entries
        loop {
            if entry.sack.contains(&entry.cumulative_ack) {
                entry.sack.remove(&entry.cumulative_ack);
                entry.cumulative_ack += 1;
            } else {
                break;
            }
        }
    }

    /// Returns `true` if `id` has been acknowledged by `receiver_id`.
    pub fn is_acked(&self, receiver_id: ReceiverId, id: SourceSymbolId) -> bool {
        self.receivers
            .get(&receiver_id)
            .is_some_and(|e| id < e.cumulative_ack || e.sack.contains(&id))
    }

    /// All known receiver IDs.
    pub fn receiver_ids(&self) -> impl Iterator<Item = ReceiverId> + '_ {
        self.receivers.keys().copied()
    }

    pub fn num_receivers(&self) -> usize {
        self.receivers.len()
    }
}

// ── WindowPolicy ──────────────────────────────────────────────────────────

/// Decides which source symbols may be removed from the encoding window
/// after a receiver acknowledges a set of IDs.
///
/// Implementations are called after [`ReceiverAckState`] is updated.
pub trait WindowPolicy: Send + 'static {
    fn symbols_to_remove(
        &mut self,
        ack_state: &ReceiverAckState,
        window_ids: &[SourceSymbolId],
    ) -> Vec<SourceSymbolId>;
}

// ── CongestionControl ────────────────────────────────────────────────────

/// Pluggable congestion control hook.
///
/// The encoder calls these methods on every send/feedback event.
/// The return values control CCI header content and send pacing.
pub trait CongestionControl: Send + 'static {
    /// Generate CCI bytes to embed in the next packet (0, 4, 8, or 12 bytes).
    fn generate_cci(&self) -> &[u8];
    /// Process received CCI bytes from a feedback packet.
    fn process_cci(&mut self, cci: &[u8]);
    /// Called when a feedback packet is received with loss statistics.
    fn on_feedback(
        &mut self,
        state: EncoderState<'_>,
        loss_rate: f64,
        nb_missing: u32,
        nb_unused_coded: u32,
    );
    /// Returns `true` when the CC algorithm permits sending a new packet.
    fn can_send(&self) -> bool;
    /// Notify the CC algorithm that a packet of `size` bytes was just sent.
    fn on_send(&mut self, size: usize);
}

// ── FecRateController ────────────────────────────────────────────────────

/// Decides how many coded packets to emit after each source submission.
pub trait FecRateController: Send + 'static {
    /// Called after a source symbol is submitted.
    ///
    /// Returns the number of coded packets the encoder should generate
    /// immediately.  Returning 0 suppresses coded output for this symbol.
    fn coded_packets_to_generate(&mut self, state: EncoderState<'_>) -> usize;

    /// Called when a feedback packet is received.
    fn on_feedback(
        &mut self,
        state: EncoderState<'_>,
        loss_rate: f64,
        nb_missing: u32,
        nb_unused: u32,
    );
}

// ── FeedbackPolicy ───────────────────────────────────────────────────────

/// Decides when the decoder should emit a `WindowUpdate` feedback packet.
pub trait FeedbackPolicy: Send + 'static {
    /// Called after every received packet.
    ///
    /// Returns `true` when the decoder should send a `WindowUpdate` now.
    fn should_send_feedback(&mut self, state: DecoderState) -> bool;
}
