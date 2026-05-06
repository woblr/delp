use thiserror::Error;

/// Every error the Delp codec can produce.
///
/// Variants are intentionally granular so callers can pattern-match on
/// specific failure modes without stringing-comparing messages.
#[derive(Debug, Error)]
pub enum DelpError {
    // ── Configuration ────────────────────────────────────────────────────
    /// `symbol_size` was zero or larger than the maximum allowed (65 535 bytes).
    #[error("symbol_size must be in 1..=65535, got {0}")]
    InvalidSymbolSize(usize),

    /// `window_capacity` was zero.
    #[error("window_capacity must be ≥ 1")]
    InvalidWindowCapacity,

    /// FEC redundancy denominator was zero (would imply infinite rate).
    #[error("fec_denom must be ≥ 1")]
    InvalidFecDenom,

    // ── Encoder ──────────────────────────────────────────────────────────
    /// Submitted payload length does not match the negotiated `symbol_size`.
    #[error("payload length {actual} does not match symbol_size {expected}")]
    SymbolSizeMismatch { expected: usize, actual: usize },

    /// Encoding window is full and backpressure policy is `Reject`.
    #[error("encoding window is full ({capacity} symbols)")]
    WindowFull { capacity: usize },

    /// Source symbol ID space exhausted (> 2^32 − 1 symbols submitted).
    #[error("source symbol ID wrapped; restart the session")]
    SourceIdExhausted,

    /// All valid coded IDs for the current matrix strategy have been used
    /// within this session.  The encoder must be reset or the session must
    /// be renegotiated before generating further coded packets.
    ///
    /// Limits:
    ///  - Vandermonde GF(2⁸): 254 unique coded IDs (1..=254)
    ///  - Vandermonde GF(2⁴): 14  unique coded IDs (1..=14)
    ///  - Cauchy GF(2⁸):      128 unique coded IDs (0..=127)
    #[error("coded ID space exhausted for this strategy/field ({used} used); restart the session")]
    CodedIdExhausted { used: u32 },

    // ── Decoder ──────────────────────────────────────────────────────────
    /// A gap opened in the delivery sequence that cannot be recovered
    /// because the encoder has already evicted those symbols from its
    /// window.  The gap starts at `first_missing`.
    #[error("unrecoverable gap starting at source symbol {first_missing}")]
    UnrecoverableGap { first_missing: u32 },

    // ── Wire format ──────────────────────────────────────────────────────
    /// The byte buffer is shorter than the minimum packet header.
    #[error("buffer too short: need {needed} bytes, got {available}")]
    BufferTooShort { needed: usize, available: usize },

    /// Version field is not 1.
    #[error("unsupported Delp protocol version {0}; expected 1")]
    UnsupportedVersion(u8),

    /// `PKT_TYPE` field carries an undefined value.
    #[error("unknown packet type 0x{0:02x}")]
    UnknownPacketType(u8),

    /// `HDR_LEN` field is inconsistent with the actual buffer length or
    /// with the minimum size for the declared packet type.
    #[error("invalid HDR_LEN {hdr_len_words} (packet is {packet_len} bytes)")]
    InvalidHeaderLength {
        hdr_len_words: u8,
        packet_len: usize,
    },

    /// `EV_LEN` field implies an encoding vector that exceeds the packet.
    #[error("encoding vector length {ev_len_words} words exceeds remaining packet bytes")]
    EncodingVectorOverflow { ev_len_words: u8 },

    /// A bit-level field in the encoding vector is inconsistent (e.g.
    /// `NB_IDS` does not match the declared ID storage format).
    #[error("malformed encoding vector: {reason}")]
    MalformedEncodingVector { reason: &'static str },

    /// `NB_COEFS` is non-zero but `C` bit is 0 (implicit coefficients
    /// are used), or vice-versa.
    #[error("coefficient field inconsistency in encoding vector")]
    CoefficientFieldInconsistency,

    /// Galois field division by zero attempted during decoding.
    #[error("GF division by zero")]
    DivisionByZero,
}

/// Convenience alias used throughout the crate.
pub type Result<T, E = DelpError> = core::result::Result<T, E>;
