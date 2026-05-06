pub mod window;

use bytes::Bytes;
use smallvec::SmallVec;
use tracing::{debug, trace};

use crate::config::MatrixStrategy;
use crate::config::{BackpressureMode, EncoderConfig, Field};
use crate::error::{DelpError, Result};
use crate::gf::simd::{dispatch, gf2_4_tbl, gf2_8_tbl, mul_acc_raw};
use crate::policy::{
    CongestionControl, EncoderState, FecRateController, ReceiverAckState, ReceiverId,
    SourceSymbolId, WindowPolicy,
};
use crate::wire::{
    coded::CodedPacket,
    ev::{
        coef_gen::{cauchy_batch_gf2_4, cauchy_batch_gf2_8, vandermonde_batch},
        EncodingVector,
    },
    feedback::FeedbackPacket,
    source::SourcePacket,
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
    config: EncoderConfig,
    window: EncodingWindow,
    /// Next coded ID to assign.  Wraps within the strategy+field range:
    /// Vandermonde 1..=ORDER-2, Cauchy 0..=127 (GF(2⁸)) / 0..=6 (GF(2⁴)).
    next_coded_id: u32,
    /// Total coded packets generated this session — informational counter.
    coded_ids_used: u64,
    /// Cauchy generation counter — increments each time `coded_id` wraps,
    /// rotating the y-point assignment so successive cycles produce
    /// linearly-independent rows.  Vandermonde always uses generation 0.
    generation: u8,
    ack_state: ReceiverAckState,
    window_policy: W,
    cc: C,
    fec: F,
    /// Reusable buffer for coded-packet payload generation (avoids per-call alloc).
    scratch: Vec<u8>,
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
        let cap = config.window_capacity;
        let first_id = match config.matrix_strategy {
            MatrixStrategy::Cauchy => 0,      // Cauchy uses 0..=127
            MatrixStrategy::Vandermonde => 1, // Vandermonde: 0 is degenerate
        };
        Self {
            config,
            window: EncodingWindow::new(cap),
            next_coded_id: first_id,
            coded_ids_used: 0,
            generation: 0,
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
                actual: data.len(),
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

        let cci = self.cc.generate_cci().to_vec();
        let src_pkt = SourcePacket::serialise(source_id, &data, &cci, None);
        self.cc.on_send(src_pkt.len());

        let state = make_encoder_state(&self.window, self.config.window_capacity);
        let n_coded = self.fec.coded_packets_to_generate(state);
        let mut out = Vec::with_capacity(1 + n_coded);
        out.push(EncoderOutput::Source(src_pkt));

        for _ in 0..n_coded {
            match self.generate_coded()? {
                Some(coded) => out.push(coded),
                None => break, // window became empty
            }
        }
        Ok(out)
    }

    /// Per-generation coded-ID cycle length for the active strategy/field.
    ///
    /// Each generation cycles through this many distinct coded IDs before
    /// wrapping; the encoder rotates the [`generation`] counter on wrap.
    ///
    /// | Strategy    | Field  | Cycle |
    /// |-------------|--------|-------|
    /// | Vandermonde | GF(2⁸) |  254  |
    /// | Vandermonde | GF(2⁴) |   14  |
    /// | Cauchy      | GF(2⁸) |  128  |
    /// | Cauchy      | GF(2⁴) |    7  |
    ///
    /// [`generation`]: Self::generation
    pub fn coded_id_cycle(&self) -> u32 {
        match self.config.matrix_strategy {
            MatrixStrategy::Cauchy => match self.config.field {
                Field::Gf2_8 => 128,
                Field::Gf2_4 => 7,
            },
            MatrixStrategy::Vandermonde => match self.config.field {
                Field::Gf2_8 => 254,
                Field::Gf2_4 => 14,
            },
        }
    }

    /// Backwards-compatible alias for [`coded_id_cycle`].
    #[deprecated(note = "use coded_id_cycle(); the per-session limit was removed")]
    pub fn coded_id_limit(&self) -> u32 {
        self.coded_id_cycle()
    }

    /// Total coded packets generated in this session (informational).
    pub fn coded_ids_used(&self) -> u64 {
        self.coded_ids_used
    }

    /// Current generation counter.  Cauchy uses this to rotate the y-point
    /// set for unlimited session length; Vandermonde keeps it at 0 and
    /// relies on the sliding window to keep wrapped coded IDs distinct.
    pub fn generation(&self) -> u8 {
        self.generation
    }

    /// Always returns `false` — delp's encoder no longer caps the session
    /// length.  Retained as a no-op for API compatibility.
    #[deprecated(note = "the per-session coded-id cap has been removed")]
    pub fn coded_id_exhausted(&self) -> bool {
        false
    }

    /// Generate one additional coded packet covering the **entire** window.
    ///
    /// Returns:
    /// - `Ok(Some(pkt))` — a coded packet ready to transmit
    /// - `Ok(None)` — window is empty, nothing to encode
    ///
    /// For targeted coding (smaller EV, less wasted redundancy on already-
    /// acknowledged symbols), use [`generate_coded_targeted`],
    /// [`generate_coded_recent`], or [`generate_coded_for_receiver`].
    ///
    /// [`generate_coded_targeted`]: Self::generate_coded_targeted
    /// [`generate_coded_recent`]:   Self::generate_coded_recent
    /// [`generate_coded_for_receiver`]: Self::generate_coded_for_receiver
    pub fn generate_coded(&mut self) -> Result<Option<EncoderOutput>> {
        if self.window.is_empty() {
            return Ok(None);
        }
        let cover: SmallVec<[u32; 64]> = self.window.symbols().iter().map(|s| s.id).collect();
        self.generate_coded_inner(cover)
    }

    /// **ALTC** — Generate a coded packet covering only the symbols in
    /// `cover_ids` (must be a subset of the current window).
    ///
    /// IDs not in the window are silently filtered out.  When the resulting
    /// cover set is empty, returns `Ok(None)`.
    ///
    /// **Why this matters:** the standard coding rule covers every symbol
    /// in the window even if most have already been delivered.  Targeting
    /// only the symbols a specific receiver still needs reduces the EV's
    /// wire size, the encoder's GF-multiplication cost, and the decoder's
    /// matrix row weight.  See [`generate_coded_for_receiver`] and
    /// [`generate_coded_recent`] for built-in selection strategies.
    ///
    /// [`generate_coded_for_receiver`]: Self::generate_coded_for_receiver
    /// [`generate_coded_recent`]:       Self::generate_coded_recent
    pub fn generate_coded_targeted(
        &mut self,
        cover_ids: &[SourceSymbolId],
    ) -> Result<Option<EncoderOutput>> {
        if self.window.is_empty() {
            return Ok(None);
        }
        let in_window: std::collections::HashSet<SourceSymbolId> =
            self.window.symbols().iter().map(|s| s.id).collect();
        let cover: SmallVec<[u32; 64]> = cover_ids
            .iter()
            .copied()
            .filter(|id| in_window.contains(id))
            .collect();
        if cover.is_empty() {
            return Ok(None);
        }
        self.generate_coded_inner(cover)
    }

    /// **ALTC** — Generate a coded packet covering only the `n` most-recent
    /// symbols in the window.  Useful for prioritising recovery of in-flight
    /// losses where retransmission is most likely to help.
    pub fn generate_coded_recent(&mut self, n: usize) -> Result<Option<EncoderOutput>> {
        if self.window.is_empty() || n == 0 {
            return Ok(None);
        }
        let syms = self.window.symbols();
        let take = n.min(syms.len());
        let cover: SmallVec<[u32; 64]> =
            syms.iter().skip(syms.len() - take).map(|s| s.id).collect();
        self.generate_coded_inner(cover)
    }

    /// **ALTC** — Generate a coded packet for a *specific receiver*: covers
    /// only window symbols that have **not yet been acknowledged** by
    /// `receiver_id` (per [`ReceiverAckState`]).
    ///
    /// If the receiver has no recorded acks the cover set falls back to the
    /// full window.  If every windowed symbol is already acknowledged,
    /// returns `Ok(None)` (no coding needed).
    pub fn generate_coded_for_receiver(
        &mut self,
        receiver_id: ReceiverId,
    ) -> Result<Option<EncoderOutput>> {
        if self.window.is_empty() {
            return Ok(None);
        }
        let cover: SmallVec<[u32; 64]> = self
            .window
            .symbols()
            .iter()
            .map(|s| s.id)
            .filter(|id| !self.ack_state.is_acked(receiver_id, *id))
            .collect();
        if cover.is_empty() {
            return Ok(None);
        }
        self.generate_coded_inner(cover)
    }

    /// Core coded-packet generator.  `cover` lists the source IDs to
    /// include — must be non-empty and a subset of the current window.
    fn generate_coded_inner(
        &mut self,
        cover: SmallVec<[u32; 64]>,
    ) -> Result<Option<EncoderOutput>> {
        debug_assert!(!cover.is_empty());

        let coded_id = self.next_coded_id;
        let generation = self.generation;

        // Advance the coded-id counter and rotate the generation on wrap
        // (Cauchy only).  See the docs on `coded_id_cycle()` for the math.
        self.next_coded_id += 1;
        self.coded_ids_used += 1;
        match self.config.matrix_strategy {
            MatrixStrategy::Vandermonde => {
                let order_minus_1 = match self.config.field {
                    Field::Gf2_8 => 255u32,
                    Field::Gf2_4 => 15u32,
                };
                if self.next_coded_id >= order_minus_1 {
                    self.next_coded_id = 1;
                }
            }
            MatrixStrategy::Cauchy => {
                let cycle = self.coded_id_cycle();
                if self.next_coded_id >= cycle {
                    self.next_coded_id = 0;
                    self.generation = self.generation.wrapping_add(1);
                }
            }
        }

        let (coefs, ev) = match self.config.matrix_strategy {
            MatrixStrategy::Vandermonde => {
                let c = vandermonde_batch(self.config.field, &cover, coded_id);
                let e = EncodingVector::vandermonde(self.config.field, coded_id, cover.clone());
                (c, e)
            }
            MatrixStrategy::Cauchy => {
                debug_assert!(
                    coded_id < self.coded_id_cycle(),
                    "coded_id {coded_id} out of Cauchy cycle range"
                );
                let c = match self.config.field {
                    Field::Gf2_8 => cauchy_batch_gf2_8(&cover, coded_id, generation),
                    Field::Gf2_4 => cauchy_batch_gf2_4(&cover, coded_id, generation),
                };
                let e = EncodingVector::explicit(
                    self.config.field,
                    coded_id,
                    cover.clone(),
                    c.iter().copied().collect(),
                )
                .with_generation(generation);
                (c, e)
            }
        };

        let payload = self.compute_coded_payload_targeted(&cover, &coefs);

        let cci = self.cc.generate_cci().to_vec();
        let pkt = CodedPacket::serialise(coded_id, &ev, &payload, &cci, None);
        self.cc.on_send(pkt.len());

        debug!(coded_id, cover_size = cover.len(),
            strategy = ?self.config.matrix_strategy, "generated coded packet");
        Ok(Some(EncoderOutput::Coded(pkt)))
    }

    /// Process a received feedback (window update) packet from a decoder.
    pub fn handle_feedback(&mut self, receiver_id: ReceiverId, pkt: &FeedbackPacket) {
        self.ack_state.update(receiver_id, pkt);

        let ids = self.window.id_slice();
        let to_rm = self.window_policy.symbols_to_remove(&self.ack_state, &ids);
        if !to_rm.is_empty() {
            debug!(?to_rm, "evicting ACK'd symbols from window");
            self.window.remove_ids(&to_rm);
        }

        let loss = FeedbackPacket::decode_plr(pkt.plr_raw);
        let nb_miss = pkt.nb_missing_src;
        let nb_unused = pkt.nb_not_used_coded;

        let state = make_encoder_state(&self.window, self.config.window_capacity);
        self.fec.on_feedback(state, loss, nb_miss, nb_unused);

        let state = make_encoder_state(&self.window, self.config.window_capacity);
        self.cc.on_feedback(state, loss, nb_miss, nb_unused);
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    pub fn window_size(&self) -> usize {
        self.window.len()
    }
    pub fn window_capacity(&self) -> usize {
        self.config.window_capacity
    }
    pub fn next_source_id(&self) -> u32 {
        self.window.next_id
    }
    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }

    // ── Internals ─────────────────────────────────────────────────────────

    /// Compute the coded payload over a targeted cover set.
    ///
    /// `cover` is the ordered list of source IDs to combine; `coefs[i]`
    /// is the coefficient for `cover[i]`.  Source IDs must be in the
    /// current window (caller is expected to have verified this).
    fn compute_coded_payload_targeted(
        &mut self,
        cover: &[SourceSymbolId],
        coefs: &[u8],
    ) -> Vec<u8> {
        let sz = self.config.symbol_size;
        let disp = dispatch();
        self.scratch.fill(0);

        match self.config.field {
            Field::Gf2_8 => {
                let tbl = gf2_8_tbl();
                for (&id, &coef) in cover.iter().zip(coefs.iter()) {
                    if coef == 0 {
                        continue;
                    }
                    let data = self.window.get(id).expect("cover_id must be in window");
                    debug_assert_eq!(data.len(), sz);
                    mul_acc_raw(&mut self.scratch, data, &tbl[coef as usize], disp);
                }
            }
            Field::Gf2_4 => {
                let tbl = gf2_4_tbl();
                for (&id, &coef) in cover.iter().zip(coefs.iter()) {
                    let c = coef & 0x0F;
                    if c == 0 {
                        continue;
                    }
                    let data = self.window.get(id).expect("cover_id must be in window");
                    debug_assert_eq!(data.len(), sz);
                    // Broadcast the 4-bit scalar across both nibbles so the
                    // SIMD split-nibble table multiplies high *and* low halves
                    // of every payload byte by `c`.
                    let packed = (c << 4) | c;
                    mul_acc_raw(&mut self.scratch, data, &tbl[packed as usize], disp);
                }
            }
        }

        self.scratch.clone()
    }

    /// Backwards-compatible alias: full-window coded payload.
    #[allow(dead_code)]
    fn compute_coded_payload(&mut self, source_ids: &[u32], coefs: &[u8]) -> Vec<u8> {
        self.compute_coded_payload_targeted(source_ids, coefs)
    }

    /// Construct a read-only state snapshot for policy callbacks.
    ///
    /// Uses a standalone function to avoid conflicting borrows of `self`.
    #[allow(dead_code)]
    fn make_state(&self) -> EncoderState<'_> {
        make_encoder_state(&self.window, self.config.window_capacity)
    }
}

/// Free function to build an `EncoderState` snapshot without borrowing
/// the policy fields of `Encoder` — avoids simultaneous mutable + immutable
/// borrow of `self`.
fn make_encoder_state<'a>(window: &'a EncodingWindow, capacity: usize) -> EncoderState<'a> {
    EncoderState {
        window_size: window.len(),
        window_capacity: capacity,
        next_source_id: window.next_id,
        next_coded_id: 0, // updated by caller if needed
        loss_rate: 0.0,
        window_ids: &[], // owned ids not available here; use window.id_slice() when needed
    }
}

// ── Convenience type alias ────────────────────────────────────────────────

use crate::policy::defaults::{AnyAckPolicy, ConstantFecRate, NoCongestionControl};

/// A ready-to-use encoder with sensible zero-config defaults.
///
/// - `AnyAckPolicy` — remove symbols on first ACK (unicast-optimal)
/// - `NoCongestionControl` — no pacing, no CCI
/// - `ConstantFecRate` — derived from the config's `fec_numer`/`fec_denom`
///   (defaults to 1:4 = 25 % overhead when the builder default is used)
pub type DefaultEncoder = Encoder<AnyAckPolicy, NoCongestionControl, ConstantFecRate>;

impl DefaultEncoder {
    pub fn with_defaults(config: EncoderConfig) -> Self {
        let fec = ConstantFecRate::new(config.fec_numer, config.fec_denom);
        Encoder::new(config, AnyAckPolicy, NoCongestionControl, fec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EncoderConfig;
    use crate::policy::defaults::{AnyAckPolicy, ConstantFecRate, NoCongestionControl};

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
        let data = Bytes::from(vec![0xABu8; 64]);
        let out = enc.submit_source(data).unwrap();
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
        let mut enc = Encoder::new(
            cfg,
            AnyAckPolicy,
            NoCongestionControl,
            ConstantFecRate::new(1, 1),
        );
        let data = Bytes::from(vec![0u8; 64]);
        let out = enc.submit_source(data).unwrap();
        assert_eq!(out.len(), 2);
        assert!(matches!(out[0], EncoderOutput::Source(_)));
        assert!(matches!(out[1], EncoderOutput::Coded(_)));
    }

    #[test]
    fn symbol_size_mismatch_error() {
        let mut enc = make_encoder(64, 0, 1);
        let result = enc.submit_source(Bytes::from(vec![0u8; 32]));
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

    /// Vandermonde wraps coded_id beyond the per-cycle limit without
    /// erroring — sliding-window operation keeps wrapped IDs distinct.
    #[test]
    fn vandermonde_wraps_past_cycle_without_error() {
        let cfg = EncoderConfig::builder(8)
            .window_capacity(32)
            .fec_rate(0, 1)
            .build()
            .unwrap();
        let mut enc = DefaultEncoder::with_defaults(cfg);
        enc.submit_source(Bytes::from(vec![0xABu8; 8])).unwrap();

        let cycle = enc.coded_id_cycle();
        // Generate 3× the cycle — every call must succeed.
        for i in 0..(cycle * 3) {
            enc.generate_coded()
                .unwrap_or_else(|e| panic!("packet {i} failed: {e}"))
                .expect("window should be non-empty");
        }
        assert_eq!(enc.coded_ids_used(), (cycle * 3) as u64);
        // Vandermonde never advances the generation counter.
        assert_eq!(enc.generation(), 0);
    }

    /// Cauchy advances the generation counter when coded_id wraps,
    /// rotating the y-point set so the session can run indefinitely.
    #[test]
    fn cauchy_generation_rotates_on_wrap() {
        let cfg = EncoderConfig::builder(8)
            .window_capacity(32)
            .matrix_strategy(crate::config::MatrixStrategy::Cauchy)
            .fec_rate(0, 1)
            .build()
            .unwrap();
        let mut enc = DefaultEncoder::with_defaults(cfg);
        enc.submit_source(Bytes::from(vec![0x42u8; 8])).unwrap();

        // 127 packets — still generation 0 (next_coded_id 0..127 used).
        for _ in 0..127u32 {
            enc.generate_coded().unwrap().unwrap();
        }
        assert_eq!(
            enc.generation(),
            0,
            "first cycle should keep generation at 0"
        );

        // 128th packet uses coded_id=127, then wraps and bumps generation.
        enc.generate_coded().unwrap().unwrap();
        assert_eq!(
            enc.generation(),
            1,
            "wrap after one full cycle should bump generation to 1"
        );

        // Six more full cycles → generation reaches 7.
        for _ in 0..(128u32 * 6) {
            enc.generate_coded().unwrap().unwrap();
        }
        assert_eq!(enc.generation(), 7);
        assert_eq!(enc.coded_ids_used(), 128u64 * 7);
    }

    /// Wire-format round-trip: a Cauchy coded packet generated after
    /// generation rotation must serialise / parse cleanly with the
    /// generation byte preserved.
    #[test]
    fn cauchy_generation_survives_wire_round_trip() {
        use crate::wire::coded::CodedPacket;
        let cfg = EncoderConfig::builder(8)
            .window_capacity(32)
            .matrix_strategy(crate::config::MatrixStrategy::Cauchy)
            .fec_rate(0, 1)
            .build()
            .unwrap();
        let mut enc = DefaultEncoder::with_defaults(cfg);
        enc.submit_source(Bytes::from(vec![0x55u8; 8])).unwrap();

        // Burn through one cycle to roll generation to 1.
        for _ in 0..128u32 {
            enc.generate_coded().unwrap().unwrap();
        }
        assert_eq!(enc.generation(), 1);

        // Next coded packet carries generation=1.
        if let Some(EncoderOutput::Coded(raw)) = enc.generate_coded().unwrap() {
            let parsed = CodedPacket::parse(&raw).unwrap();
            assert_eq!(parsed.ev.generation, 1, "parsed EV must carry generation=1");
            assert!(parsed.ev.has_explicit_coefs());
        } else {
            panic!("expected Coded output");
        }
    }
}
