use super::{
    WindowPolicy, CongestionControl, FecRateController, FeedbackPolicy,
    ReceiverAckState, SourceSymbolId, EncoderState, DecoderState,
};

// ── WindowPolicy implementations ─────────────────────────────────────────

/// Remove a symbol as soon as **any** receiver acknowledges it.
///
/// Optimal for unicast; reduces window size aggressively.
/// Not safe for multicast (a slow receiver may never recover lost symbols).
#[derive(Debug, Default, Clone)]
pub struct AnyAckPolicy;

impl WindowPolicy for AnyAckPolicy {
    fn symbols_to_remove(
        &mut self,
        ack_state:  &ReceiverAckState,
        window_ids: &[SourceSymbolId],
    ) -> Vec<SourceSymbolId> {
        window_ids.iter().filter(|&&id| {
            ack_state.receiver_ids().any(|rid| ack_state.is_acked(rid, id))
        }).copied().collect()
    }
}

/// Remove a symbol only when **every** known receiver has acknowledged it.
///
/// Safe for multicast: every receiver gets a chance to recover the symbol.
/// May keep the window large when receivers have divergent ACK progress.
#[derive(Debug, Default, Clone)]
pub struct AllAckPolicy;

impl WindowPolicy for AllAckPolicy {
    fn symbols_to_remove(
        &mut self,
        ack_state:  &ReceiverAckState,
        window_ids: &[SourceSymbolId],
    ) -> Vec<SourceSymbolId> {
        if ack_state.num_receivers() == 0 { return Vec::new(); }
        window_ids.iter().filter(|&&id| {
            ack_state.receiver_ids().all(|rid| ack_state.is_acked(rid, id))
        }).copied().collect()
    }
}

/// Quorum policy: remove a symbol when at least `quorum` out of all known
/// receivers have acknowledged it.
#[derive(Debug, Clone)]
pub struct QuorumAckPolicy {
    pub quorum: usize,
}

impl QuorumAckPolicy {
    pub fn new(quorum: usize) -> Self { Self { quorum } }
}

impl WindowPolicy for QuorumAckPolicy {
    fn symbols_to_remove(
        &mut self,
        ack_state:  &ReceiverAckState,
        window_ids: &[SourceSymbolId],
    ) -> Vec<SourceSymbolId> {
        window_ids.iter().filter(|&&id| {
            let acked_count = ack_state.receiver_ids()
                .filter(|&rid| ack_state.is_acked(rid, id))
                .count();
            acked_count >= self.quorum
        }).copied().collect()
    }
}

// ── CongestionControl implementations ────────────────────────────────────

/// No congestion control — always allows sending, emits zero CCI bytes.
///
/// Suitable for controlled lab environments or when the transport layer
/// already provides congestion control (e.g. QUIC).
#[derive(Debug, Default, Clone)]
pub struct NoCongestionControl;

impl CongestionControl for NoCongestionControl {
    fn generate_cci(&self)                              -> &[u8] { &[] }
    fn process_cci(&mut self, _cci: &[u8])              {}
    fn on_feedback(&mut self, _: EncoderState<'_>, _: f64, _: u32, _: u32) {}
    fn can_send(&self)                                  -> bool { true }
    fn on_send(&mut self, _size: usize)                 {}
}

// ── FecRateController implementations ────────────────────────────────────

/// Emit exactly `numer` coded packets for every `denom` source packets.
///
/// Example: `ConstantFecRate::new(1, 4)` → 25 % overhead (1 coded per 4 source).
#[derive(Debug, Clone)]
pub struct ConstantFecRate {
    numer:   usize,
    denom:   usize,
    /// Running numerator accumulator.
    acc:     usize,
}

impl ConstantFecRate {
    /// Create a rate controller emitting `numer` coded packets per `denom` source packets.
    pub fn new(numer: usize, denom: usize) -> Self {
        assert!(denom > 0, "denom must be ≥ 1");
        Self { numer, denom, acc: 0 }
    }

    /// 1:1 ratio — one coded packet for every source packet.
    pub fn one_to_one() -> Self { Self::new(1, 1) }

    /// No coded packets emitted at all (FEC disabled).
    pub fn disabled() -> Self { Self::new(0, 1) }
}

impl FecRateController for ConstantFecRate {
    fn coded_packets_to_generate(&mut self, _state: EncoderState<'_>) -> usize {
        self.acc += self.numer;
        let count = self.acc / self.denom;
        self.acc %= self.denom;
        count
    }

    fn on_feedback(&mut self, _: EncoderState<'_>, _: f64, _: u32, _: u32) {}
}

/// Adaptive FEC rate: adjusts coded output based on observed loss rate.
///
/// When loss_rate > `target_loss` the rate is doubled (up to `max_rate`);
/// when loss_rate < `target_loss / 2` the rate is halved (down to `min_rate`).
#[derive(Debug, Clone)]
pub struct AdaptiveFecRate {
    /// Baseline FEC rate (coded per source).
    current_rate: f64,
    min_rate:     f64,
    max_rate:     f64,
    target_loss:  f64,
    /// Fractional accumulator for fractional rates.
    acc:          f64,
}

impl AdaptiveFecRate {
    pub fn new(initial_rate: f64, min_rate: f64, max_rate: f64, target_loss: f64) -> Self {
        Self {
            current_rate: initial_rate.clamp(min_rate, max_rate),
            min_rate,
            max_rate,
            target_loss,
            acc: 0.0,
        }
    }
}

impl FecRateController for AdaptiveFecRate {
    fn coded_packets_to_generate(&mut self, _state: EncoderState<'_>) -> usize {
        self.acc += self.current_rate;
        let count = self.acc.floor() as usize;
        self.acc -= count as f64;
        count
    }

    fn on_feedback(
        &mut self,
        _state:     EncoderState<'_>,
        loss_rate:  f64,
        _nb_miss:   u32,
        _nb_unused: u32,
    ) {
        if loss_rate > self.target_loss {
            self.current_rate = (self.current_rate * 2.0).min(self.max_rate);
        } else if loss_rate < self.target_loss / 2.0 {
            self.current_rate = (self.current_rate / 2.0).max(self.min_rate);
        }
    }
}

// ── FeedbackPolicy implementations ───────────────────────────────────────

/// Emit one feedback packet every `period` received packets.
#[derive(Debug, Clone)]
pub struct ConstantFeedbackPolicy {
    period:  u32,
    counter: u32,
}

impl ConstantFeedbackPolicy {
    pub fn new(period: u32) -> Self {
        assert!(period > 0, "feedback period must be ≥ 1");
        Self { period, counter: 0 }
    }
}

impl FeedbackPolicy for ConstantFeedbackPolicy {
    fn should_send_feedback(&mut self, _state: DecoderState) -> bool {
        self.counter += 1;
        if self.counter >= self.period {
            self.counter = 0;
            true
        } else {
            false
        }
    }
}

/// Emit feedback immediately after every received packet.  Maximises
/// encoder responsiveness at the cost of reverse-path overhead.
#[derive(Debug, Default, Clone)]
pub struct ImmediateFeedbackPolicy;

impl FeedbackPolicy for ImmediateFeedbackPolicy {
    fn should_send_feedback(&mut self, _state: DecoderState) -> bool { true }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_state() -> EncoderState<'static> {
        EncoderState {
            window_size:     4,
            window_capacity: 256,
            next_source_id:  10,
            next_coded_id:   3,
            loss_rate:       0.0,
            window_ids:      &[],
        }
    }

    #[test]
    fn constant_fec_rate_one_to_one() {
        let mut ctrl = ConstantFecRate::one_to_one();
        for _ in 0..8 {
            assert_eq!(ctrl.coded_packets_to_generate(dummy_state()), 1);
        }
    }

    #[test]
    fn constant_fec_rate_one_in_four() {
        let mut ctrl = ConstantFecRate::new(1, 4);
        let counts: Vec<usize> = (0..8).map(|_| ctrl.coded_packets_to_generate(dummy_state())).collect();
        // Expect 1 coded per 4 source on average
        assert_eq!(counts.iter().sum::<usize>(), 2); // 8 / 4 = 2
    }

    #[test]
    fn adaptive_rate_increases_on_loss() {
        let mut ctrl = AdaptiveFecRate::new(0.25, 0.1, 1.0, 0.05);
        let state = dummy_state();
        ctrl.on_feedback(state, 0.2, 5, 0); // loss > target → rate doubles
        assert!(ctrl.current_rate > 0.25);
    }

    #[test]
    fn adaptive_rate_decreases_on_low_loss() {
        let mut ctrl = AdaptiveFecRate::new(0.5, 0.1, 1.0, 0.05);
        let state = dummy_state();
        ctrl.on_feedback(state, 0.0, 0, 0); // loss < target/2 → rate halves
        assert!(ctrl.current_rate < 0.5);
    }

    #[test]
    fn any_ack_policy_removes_on_first_ack() {
        use crate::wire::feedback::FeedbackPacket;
        let mut state  = ReceiverAckState::default();
        let acked       = vec![10u32, 11, 12];
        let pkt         = FeedbackPacket::build(10, 0, 0, 0.0, &acked);
        state.update(1, &pkt);
        let mut policy  = AnyAckPolicy;
        let window: Vec<u32> = (10..15).collect();
        let to_remove   = policy.symbols_to_remove(&state, &window);
        assert!(to_remove.contains(&10));
        assert!(to_remove.contains(&11));
        assert!(to_remove.contains(&12));
        assert!(!to_remove.contains(&13));
    }

    #[test]
    fn all_ack_policy_requires_all_receivers() {
        use crate::wire::feedback::FeedbackPacket;
        let mut state    = ReceiverAckState::default();
        let pkt1         = FeedbackPacket::build(10, 0, 0, 0.0, &[10, 11]);
        let pkt2         = FeedbackPacket::build(10, 0, 0, 0.0, &[10]);
        state.update(1, &pkt1);
        state.update(2, &pkt2);
        let mut policy   = AllAckPolicy;
        let window: Vec<u32> = (10..13).collect();
        let to_remove    = policy.symbols_to_remove(&state, &window);
        assert!( to_remove.contains(&10)); // acked by both
        assert!(!to_remove.contains(&11)); // only receiver 1
        assert!(!to_remove.contains(&12)); // neither
    }
}