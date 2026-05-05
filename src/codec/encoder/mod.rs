pub mod window;

use bytes::Bytes;
use smallvec::SmallVec;
use tracing::{debug, trace};

use crate::config::{EncoderConfig, BackpressureMode, Field};
use crate::error::{Result, DelpError};
use crate::policy::{
    WindowPolicy, CongestionControl, FecRateController,
    ReceiverAckState, ReceiverId, EncoderState,
};
use crate::config::MatrixStrategy;
use crate::gf::simd::{dispatch, gf2_8_tbl, gf2_4_tbl, mul_acc_raw};
use crate::wire::{
    feedback::FeedbackPacket,
    source::SourcePacket,
    coded::CodedPacket,
    ev::{EncodingVector, coef_gen::{vandermonde_batch, cauchy_batch_gf2_8, cauchy_batch_gf2_4}},
};
use window::EncodingWindow;

// ── Output types ─────────────────────────────────────────────────────────

/// One unit of output from the encoder — either a source packet or a
/// coded (FEC) packet, both ready to transmit as-is.
#[derive(Debug, Clone)]
pub enum EncoderOutput {
    Source(Vec<u8>),
    Coded(Vec<u8>),
}

// ── Encoder ───────────────────────────────────────────────────────────────

/// RFC 9407 Delp encoder state machine.
///
/// Generic over three pluggable policy traits:
/// - `W`: [`WindowPolicy`]       — controls window eviction after ACKs
/// - `C`: [`CongestionControl`]  — provides CCI and pacing
/// - `F`: [`FecRateController`]  — decides how many coded packets to emit
///
/// The encoder is a pure state machine: it has no I/O, no threads, no async.
/// All methods take `&mut self` and return data synchronously.
///
/// # Example
/// ```ignore
/// let cfg = EncoderConfig::builder(1024).fec_rate(1, 4).build().unwrap();
/// let mut enc = Encoder::new(cfg, AnyAckPolicy, NoCongestionControl, ConstantFecRate::new(1,4));
/// let output = enc.submit_source(Bytes::from(vec![0u8; 1024])).unwrap();
/// ```
pub struct Encoder<W, C, F>
where
    W: WindowPolicy,
    C: CongestionControl,
    F: FecRateController,
{
    config:        EncoderConfig,
    window:        EncodingWindow,
    /// Next coded ID to assign.  Never zero; wraps within the safe range
    /// for the current strategy+field combination.  See `coded_id_limit()`.
    next_coded_id: u32,
    /// Total coded packets generated this session — used for exhaustion check.
    coded_ids_used: u32,
    ack_state:     ReceiverAckState,
    window_policy: W,
    cc:            C,
    fec:           F,
    /// Reusable buffer for coded-packet payload generation (avoids per-call alloc).
    scratch:       Vec<u8>,
}

impl<W, C, F> Encoder<W, C, F>
where
    W: WindowPolicy,
    C: CongestionControl,
    F: FecRateController,
{
    // ── Construction ─────────────────────────────────────────────────────

    pub fn new(config: EncoderConfig, window_policy: W, cc: C, fec: F) -> Self {
        let scratch = vec![0u8; config.symbol_size];
        let cap     = config.window_capacity;
        let first_id = match config.matrix_strategy {
            MatrixStrategy::Cauchy      => 0, // Cauchy uses 0..=127
            MatrixStrategy::Vandermonde => 1, // Vandermonde: 0 is degenerate
        };
        Self {
            config,
            window: EncodingWindow::new(cap),
            next_coded_id: first_id,
            coded_ids_used: 0,
            ack_state: ReceiverAckState::default(),
            window_policy,
            cc,
            fec,
            scratch,
        }
    }

    // ── Public API ────────────────────────────────────────────────────────

    /// Submit one source symbol.
    ///
    /// Returns a `Vec` of serialised packets ready to send: always at least
    /// one `Source` packet, followed by zero or more `Coded` packets as
    /// determined by the [`FecRateController`].
    pub fn submit_source(&mut self, data: Bytes) -> Result<Vec<EncoderOutput>> {
        if data.len() != self.config.symbol_size {
            return Err(DelpError::SymbolSizeMismatch {
                expected: self.config.symbol_size,
                actual:   data.len(),
            });
        }

        // Backpressure
        if self.window.is_full() {
            match self.config.backpressure {
                BackpressureMode::Reject => {
                    return Err(DelpError::WindowFull {
                        capacity: self.config.window_capacity,
                    });
                }
                BackpressureMode::EvictOldest => {
                    self.window.evict_oldest();
                }
            }
        }

        let source_id = self.window.push(data.clone());
        trace!(source_id, "submitted source symbol");

        let cci      = self.cc.generate_cci().to_vec();
        let src_pkt  = SourcePacket::serialise(source_id, &data, &cci, None);
        self.cc.on_send(src_pkt.len());

        let state    = make_encoder_state(&self.window, self.config.window_capacity);
        let n_coded  = self.fec.coded_packets_to_generate(state);
        let mut out  = Vec::with_capacity(1 + n_coded);
        out.push(EncoderOutput::Source(src_pkt));

        for _ in 0..n_coded {
            match self.generate_coded()? {
                Some(coded) => out.push(coded),
                None => break, // window became empty
            }
        }
        Ok(out)
    }

    /// Maximum number of distinct coded IDs allowed per session.
    ///
    /// | Strategy    | Field  | Limit |
    /// |-------------|--------|-------|
    /// | Vandermonde | GF(2⁸) |  254  |
    /// | Vandermonde | GF(2⁴) |   14  |
    /// | Cauchy      | GF(2⁸) |  128  |
    pub fn coded_id_limit(&self) -> u32 {
        match self.config.matrix_strategy {
            MatrixStrategy::Cauchy => match self.config.field {
                Field::Gf2_8 => 128, // 0..=127
                Field::Gf2_4 => 7,   // 0..=6
            },
            MatrixStrategy::Vandermonde => match self.config.field {
                Field::Gf2_8 => 254, // 1..=254; 0 and 255 are degenerate
                Field::Gf2_4 => 14,  // 1..=14
            },
        }
    }

    /// How many coded packets have been generated in this session.
    pub fn coded_ids_used(&self) -> u32 { self.coded_ids_used }

    /// True when the coded ID space is exhausted for this session.
    /// After this returns `true`, `generate_coded` will return
    /// `Err(CodedIdExhausted)`.
    pub fn coded_id_exhausted(&self) -> bool {
        self.coded_ids_used >= self.coded_id_limit()
    }

    /// Generate one additional coded packet from the current window contents.
    ///
    /// Returns:
    /// - `Ok(Some(pkt))` — a coded packet ready to transmit
    /// - `Ok(None)` — window is empty, nothing to encode
    /// - `Err(CodedIdExhausted)` — all valid coded IDs used; restart session
    pub fn generate_coded(&mut self) -> Result<Option<EncoderOutput>> {
        if self.window.is_empty() { return Ok(None); }

        // Guard: coded_id space exhausted for this strategy+field
        if self.coded_ids_used >= self.coded_id_limit() {
            return Err(DelpError::CodedIdExhausted {
                used: self.coded_ids_used,
            });
        }

        let coded_id = self.next_coded_id;

        // Advance the counter within the safe range for this strategy.
        // Vandermonde: 1..=254 (GF2⁸) or 1..=14 (GF2⁴)
        // Cauchy:      0..=127
        self.next_coded_id += 1;
        self.coded_ids_used += 1;
        // Cauchy stays in 0..=127 — checked by coded_ids_used guard above.
        // Vandermonde wraps within the safe range, skip 0 and ORDER-1.
        if self.config.matrix_strategy == MatrixStrategy::Vandermonde {
            let order_minus_1 = match self.config.field {
                Field::Gf2_8 => 255u32,
                Field::Gf2_4 => 15u32,
            };
            if self.next_coded_id >= order_minus_1 {
                self.next_coded_id = 1; // wrap: skip 0 (degenerate) and ORDER-1
            }
        }

        let source_ids: SmallVec<[u32; 64]> =
            self.window.symbols().iter().map(|s| s.id).collect();

        let (coefs, ev) = match self.config.matrix_strategy {
            MatrixStrategy::Vandermonde => {
                let c = vandermonde_batch(self.config.field, &source_ids, coded_id);
                let e = EncodingVector::vandermonde(self.config.field, coded_id, source_ids.clone());
                (c, e)
            }
            MatrixStrategy::Cauchy => {
                // coded_ids_used guard above ensures coded_id is within safe range.
                // Builder enforces field=GF(2^8)→limit 128, field=GF(2^4)→limit 7.
                debug_assert!(coded_id < self.coded_id_limit(),
                    "coded_id {coded_id} exceeds Cauchy limit — enforced by guard");
                let c = match self.config.field {
                    Field::Gf2_8 => cauchy_batch_gf2_8(&source_ids, coded_id),
                    Field::Gf2_4 => cauchy_batch_gf2_4(&source_ids, coded_id),
                };
                // Encode coefficients explicitly — the receiver has no formula
                // to re-derive Cauchy coefs from (src_id, coded_id) alone.
                let e = EncodingVector::explicit(
                    self.config.field,
                    coded_id,
                    source_ids.clone(),
                    c.iter().copied().collect(),
                );
                (c, e)
            }
        };

        let payload = self.compute_coded_payload(&source_ids, &coefs);

        let cci = self.cc.generate_cci().to_vec();
        let pkt = CodedPacket::serialise(coded_id, &ev, &payload, &cci, None);
        self.cc.on_send(pkt.len());

        debug!(coded_id, strategy = ?self.config.matrix_strategy, "generated coded packet");
        Ok(Some(EncoderOutput::Coded(pkt)))
    }

    /// Process a received feedback (window update) packet from a decoder.
    pub fn handle_feedback(&mut self, receiver_id: ReceiverId, pkt: &FeedbackPacket) {
        self.ack_state.update(receiver_id, pkt);

        let ids     = self.window.id_slice();
        let to_rm   = self.window_policy.symbols_to_remove(&self.ack_state, &ids);
        if !to_rm.is_empty() {
            debug!(?to_rm, "evicting ACK'd symbols from window");
            self.window.remove_ids(&to_rm);
        }

        let loss  = FeedbackPacket::decode_plr(pkt.plr_raw);
        let nb_miss   = pkt.nb_missing_src;
        let nb_unused = pkt.nb_not_used_coded;

        let state = make_encoder_state(&self.window, self.config.window_capacity);
        self.fec.on_feedback(state, loss, nb_miss, nb_unused);

        let state = make_encoder_state(&self.window, self.config.window_capacity);
        self.cc.on_feedback(state, loss, nb_miss, nb_unused);
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    pub fn window_size(&self)     -> usize { self.window.len() }
    pub fn window_capacity(&self) -> usize { self.config.window_capacity }
    pub fn next_source_id(&self)  -> u32   { self.window.next_id }
    pub fn config(&self) -> &EncoderConfig { &self.config }

    // ── Internals ─────────────────────────────────────────────────────────

    /// Compute the coded payload: `sum over window of (coef_i * symbol_i)`.
    ///
    /// Hoists the SIMD dispatch check and nibble-table pointer out of the
    /// inner loop so they are resolved exactly once per coded packet rather
    /// than once per source symbol.
    fn compute_coded_payload(&mut self, _source_ids: &[u32], coefs: &[u8]) -> Vec<u8> {
        let sz   = self.config.symbol_size;
        let disp = dispatch();
        self.scratch.fill(0);

        match self.config.field {
            Field::Gf2_8 => {
                let tbl = gf2_8_tbl();
                for (sym, &coef) in self.window.symbols().iter().zip(coefs.iter()) {
                    debug_assert_eq!(sym.data.len(), sz);
                    if coef == 0 { continue; }
                    mul_acc_raw(&mut self.scratch, &sym.data, &tbl[coef as usize], disp);
                }
            }
            Field::Gf2_4 => {
                let tbl = gf2_4_tbl();
                for (sym, &coef) in self.window.symbols().iter().zip(coefs.iter()) {
                    debug_assert_eq!(sym.data.len(), sz);
                    if coef == 0 { continue; }
                    mul_acc_raw(&mut self.scratch, &sym.data, &tbl[coef as usize], disp);
                }
            }
        }

        self.scratch.clone()
    }

    /// Construct a read-only state snapshot for policy callbacks.
    ///
    /// Uses a standalone function to avoid conflicting borrows of `self`.
    #[allow(dead_code)]
    fn make_state(&self) -> EncoderState<'_> {
        make_encoder_state(
            &self.window,
            self.config.window_capacity,
        )
    }
}

/// Free function to build an `EncoderState` snapshot without borrowing
/// the policy fields of `Encoder` — avoids simultaneous mutable + immutable
/// borrow of `self`.
fn make_encoder_state<'a>(window: &'a EncodingWindow, capacity: usize) -> EncoderState<'a> {
    EncoderState {
        window_size:     window.len(),
        window_capacity: capacity,
        next_source_id:  window.next_id,
        next_coded_id:   0, // updated by caller if needed
        loss_rate:       0.0,
        window_ids:      &[], // owned ids not available here; use window.id_slice() when needed
    }
}

// ── Convenience type alias ────────────────────────────────────────────────

use crate::policy::defaults::{AnyAckPolicy, NoCongestionControl, ConstantFecRate};

/// A ready-to-use encoder with sensible zero-config defaults.
///
/// - `AnyAckPolicy` — remove symbols on first ACK (unicast-optimal)
/// - `NoCongestionControl` — no pacing, no CCI
/// - `ConstantFecRate(1, 4)` — 25 % coded overhead
pub type DefaultEncoder = Encoder<AnyAckPolicy, NoCongestionControl, ConstantFecRate>;

impl DefaultEncoder {
    pub fn with_defaults(config: EncoderConfig) -> Self {
        Encoder::new(
            config,
            AnyAckPolicy,
            NoCongestionControl,
            ConstantFecRate::new(1, 4),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EncoderConfig;
    use crate::policy::defaults::{AnyAckPolicy, NoCongestionControl, ConstantFecRate};

    fn make_encoder(sym_size: usize, fec_numer: usize, fec_denom: usize) -> DefaultEncoder {
        let cfg = EncoderConfig::builder(sym_size)
            .window_capacity(32)
            .fec_rate(fec_numer, fec_denom)
            .build()
            .unwrap();
        DefaultEncoder::with_defaults(cfg)
    }

    #[test]
    fn submit_returns_source_packet() {
        let mut enc = make_encoder(64, 0, 1); // FEC disabled
        let data    = Bytes::from(vec![0xABu8; 64]);
        let out     = enc.submit_source(data).unwrap();
        assert_eq!(out.len(), 1);
        assert!(matches!(out[0], EncoderOutput::Source(_)));
    }

    #[test]
    fn submit_returns_coded_packet_at_rate() {
        // Build encoder explicitly with 1:1 FEC rate (not using DefaultEncoder::with_defaults)
        let cfg = EncoderConfig::builder(64)
            .window_capacity(32)
            .fec_rate(1, 1)
            .build()
            .unwrap();
        let mut enc = Encoder::new(cfg, AnyAckPolicy, NoCongestionControl, ConstantFecRate::new(1, 1));
        let data = Bytes::from(vec![0u8; 64]);
        let out  = enc.submit_source(data).unwrap();
        assert_eq!(out.len(), 2);
        assert!(matches!(out[0], EncoderOutput::Source(_)));
        assert!(matches!(out[1], EncoderOutput::Coded(_)));
    }

    #[test]
    fn symbol_size_mismatch_error() {
        let mut enc = make_encoder(64, 0, 1);
        let result  = enc.submit_source(Bytes::from(vec![0u8; 32]));
        assert!(matches!(result, Err(DelpError::SymbolSizeMismatch { .. })));
    }

    #[test]
    fn window_full_reject() {
        let cfg = EncoderConfig::builder(8)
            .window_capacity(2)
            .fec_rate(0, 1)
            .build()
            .unwrap();
        let mut enc = DefaultEncoder::with_defaults(cfg);
        enc.submit_source(Bytes::from(vec![1u8; 8])).unwrap();
        enc.submit_source(Bytes::from(vec![2u8; 8])).unwrap();
        let result = enc.submit_source(Bytes::from(vec![3u8; 8]));
        assert!(matches!(result, Err(DelpError::WindowFull { .. })));
    }

    #[test]
    fn coded_payload_length_matches_symbol_size() {
        let mut enc = make_encoder(128, 1, 1);
        for _ in 0..4 {
            enc.submit_source(Bytes::from(vec![0xFFu8; 128])).unwrap();
        }
        if let Ok(Some(EncoderOutput::Coded(pkt))) = enc.generate_coded() {
            assert!(!pkt.is_empty());
        }
    }

    /// coded_id space exhausts after exactly `coded_id_limit()` packets.
    #[test]
    fn coded_id_exhaustion_returns_error() {
        // Vandermonde GF(2^8) limit = 254
        let cfg = EncoderConfig::builder(8)
            .window_capacity(32)
            .fec_rate(0, 1)
            .build()
            .unwrap();
        let mut enc = DefaultEncoder::with_defaults(cfg);
        // Push one source so the window is non-empty
        enc.submit_source(Bytes::from(vec![0xABu8; 8])).unwrap();

        let limit = enc.coded_id_limit();
        // Generate exactly `limit` coded packets — all should succeed
        for i in 0..limit {
            enc.generate_coded()
                .unwrap_or_else(|e| panic!("packet {i} failed: {e}"))
                .expect("window should be non-empty");
        }
        // The very next call must return CodedIdExhausted
        assert!(matches!(
            enc.generate_coded(),
            Err(DelpError::CodedIdExhausted { .. })
        ), "expected CodedIdExhausted after {limit} packets");
    }

    /// Cauchy strategy exhausts after exactly 128 coded packets.
    #[test]
    fn cauchy_coded_id_exhaustion() {
        let cfg = EncoderConfig::builder(8)
            .window_capacity(32)
            .matrix_strategy(crate::config::MatrixStrategy::Cauchy)
            .fec_rate(0, 1)
            .build()
            .unwrap();
        let mut enc = DefaultEncoder::with_defaults(cfg);
        enc.submit_source(Bytes::from(vec![0x42u8; 8])).unwrap();

        for i in 0..128u32 {
            enc.generate_coded()
                .unwrap_or_else(|e| panic!("Cauchy packet {i} failed: {e}"))
                .expect("window non-empty");
        }
        assert!(matches!(
            enc.generate_coded(),
            Err(DelpError::CodedIdExhausted { .. })
        ), "Cauchy: expected exhaustion after 128 packets");
    }
}