use crate::error::{DelpError, Result};

/// Which Galois field to use for coefficient arithmetic.
///
/// - `Gf2_8` — 8-bit coefficients, lower overhead per symbol, wider window.
/// - `Gf2_4` — 4-bit coefficients, two per byte; useful for very small
///   symbol sizes where header overhead dominates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Field {
    #[default]
    Gf2_8,
    Gf2_4,
}

/// Which matrix construction to use for generating coded-packet coefficients.
///
/// Both strategies produce valid FEC — the difference is in their
/// mathematical guarantees and parameter constraints.
///
/// | Strategy    | MDS guarantee | Max window | Notes                        |
/// |-------------|---------------|------------|------------------------------|
/// | Vandermonde | none¹         | 254 (GF2⁸) | lower per-coef cost          |
/// | Cauchy      | **proven**    | 128 (GF2⁸) | every k×k submatrix full rank|
///
/// ¹ Vandermonde can produce linearly-dependent rows when
///   `src_id * coded_id mod 254` collides across different (src, coded) pairs.
///
/// **Default:** `Vandermonde` — compatible with the original Delp protocol.
/// Use `Cauchy` when you need guaranteed recovery from any k-erasure pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MatrixStrategy {
    /// `coef(src, coded) = alpha ^ (src * coded mod (ORDER-1))`
    ///
    /// Standard Delp/RFC 9407 construction.  Fast to compute; no strict
    /// MDS proof for all window sizes.
    #[default]
    Vandermonde,

    /// `coef(src, coded) = 1 / (src XOR (128 + coded))` in GF(2⁸)
    ///
    /// Every square submatrix is provably full rank.  Requires
    /// `src_id < 128` and `coded_id < 128`; enforced at runtime.
    Cauchy,
}

/// What the encoder should do when [`submit_source`] is called while the
/// encoding window is already at `window_capacity`.
///
/// [`submit_source`]: crate::codec::encoder::Encoder::submit_source
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackpressureMode {
    /// Return [`DelpError::WindowFull`] — the caller must retry or apply
    /// application-level backpressure.
    #[default]
    Reject,
    /// Silently evict the oldest symbol from the window and proceed.
    /// The evicted symbol can no longer be recovered by any receiver.
    EvictOldest,
}

/// Immutable configuration shared by an encoder and all its paired decoders.
///
/// Constructed via [`EncoderConfig::builder`].
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    /// Fixed byte length of every source symbol.
    pub(crate) symbol_size: usize,
    /// Maximum number of source symbols kept in the encoding window.
    pub(crate) window_capacity: usize,
    /// Which finite field to use for coefficient arithmetic.
    pub(crate) field: Field,
    /// Numerator of the base FEC rate (can be overridden by a
    /// [`FecRateController`] policy at runtime).
    ///
    /// [`FecRateController`]: crate::policy::FecRateController
    pub(crate) fec_numer: usize,
    /// Denominator of the base FEC rate.
    pub(crate) fec_denom: usize,
    /// Backpressure behaviour when the window is full.
    pub(crate) backpressure: BackpressureMode,
    /// Which matrix construction to use for coefficient generation.
    pub(crate) matrix_strategy: MatrixStrategy,
}

impl EncoderConfig {
    /// Start building a new configuration.
    ///
    /// # Example
    /// ```
    /// use delp::config::{EncoderConfig, Field};
    ///
    /// let cfg = EncoderConfig::builder(1024)
    ///     .window_capacity(512)
    ///     .field(Field::Gf2_8)
    ///     .fec_rate(1, 4)   // 1 coded packet every 4 source packets
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn builder(symbol_size: usize) -> EncoderConfigBuilder {
        EncoderConfigBuilder::new(symbol_size)
    }

    pub fn symbol_size(&self) -> usize {
        self.symbol_size
    }
    pub fn window_capacity(&self) -> usize {
        self.window_capacity
    }
    pub fn field(&self) -> Field {
        self.field
    }
    pub fn fec_numer(&self) -> usize {
        self.fec_numer
    }
    pub fn fec_denom(&self) -> usize {
        self.fec_denom
    }
    pub fn backpressure(&self) -> BackpressureMode {
        self.backpressure
    }
    pub fn matrix_strategy(&self) -> MatrixStrategy {
        self.matrix_strategy
    }
}

/// Builder for [`EncoderConfig`].
#[derive(Debug)]
pub struct EncoderConfigBuilder {
    symbol_size: usize,
    window_capacity: usize,
    field: Field,
    fec_numer: usize,
    fec_denom: usize,
    backpressure: BackpressureMode,
    matrix_strategy: MatrixStrategy,
}

impl EncoderConfigBuilder {
    fn new(symbol_size: usize) -> Self {
        Self {
            symbol_size,
            window_capacity: 256,
            field: Field::Gf2_8,
            fec_numer: 1,
            fec_denom: 4,
            backpressure: BackpressureMode::Reject,
            matrix_strategy: MatrixStrategy::Vandermonde,
        }
    }

    /// Maximum source symbols in the encoding window (default: 256).
    pub fn window_capacity(mut self, n: usize) -> Self {
        self.window_capacity = n;
        self
    }

    /// Galois field for coefficient arithmetic (default: `Gf2_8`).
    pub fn field(mut self, f: Field) -> Self {
        self.field = f;
        self
    }

    /// Base FEC redundancy as a fraction `numer / denom`.
    ///
    /// E.g. `fec_rate(1, 4)` means one coded packet per four source packets.
    /// The runtime [`FecRateController`] policy may override this per-symbol.
    ///
    /// [`FecRateController`]: crate::policy::FecRateController
    pub fn fec_rate(mut self, numer: usize, denom: usize) -> Self {
        self.fec_numer = numer;
        self.fec_denom = denom;
        self
    }

    /// Backpressure mode when window is full (default: `Reject`).
    pub fn backpressure(mut self, mode: BackpressureMode) -> Self {
        self.backpressure = mode;
        self
    }

    /// Coefficient matrix strategy (default: `Vandermonde`).
    ///
    /// Use `Cauchy` for a mathematically proven MDS code; requires
    /// `window_capacity ≤ 128` and `coded_id < 128`.
    pub fn matrix_strategy(mut self, s: MatrixStrategy) -> Self {
        self.matrix_strategy = s;
        self
    }

    /// Validate and build the [`EncoderConfig`].
    pub fn build(self) -> Result<EncoderConfig> {
        if self.symbol_size == 0 || self.symbol_size > 65_535 {
            return Err(DelpError::InvalidSymbolSize(self.symbol_size));
        }
        if self.window_capacity == 0 {
            return Err(DelpError::InvalidWindowCapacity);
        }
        if self.fec_denom == 0 {
            return Err(DelpError::InvalidFecDenom);
        }
        if self.matrix_strategy == MatrixStrategy::Cauchy {
            let cauchy_max = match self.field {
                Field::Gf2_8 => 128,
                Field::Gf2_4 => 7,
            };
            if self.window_capacity > cauchy_max {
                return Err(DelpError::InvalidWindowCapacity);
            }
        }
        Ok(EncoderConfig {
            symbol_size: self.symbol_size,
            window_capacity: self.window_capacity,
            field: self.field,
            fec_numer: self.fec_numer,
            fec_denom: self.fec_denom,
            backpressure: self.backpressure,
            matrix_strategy: self.matrix_strategy,
        })
    }
}

/// Immutable configuration for a decoder instance.
///
/// The `symbol_size` and `field` **must** match those used by the paired
/// encoder; mismatches produce silent data corruption.
#[derive(Debug, Clone)]
pub struct DecoderConfig {
    pub(crate) symbol_size: usize,
    pub(crate) field: Field,
    /// Maximum number of coded equations held simultaneously in the
    /// decoding matrix.  Rows beyond this limit are discarded.
    pub(crate) max_matrix_rows: usize,
    /// Decoder emits a `WindowUpdate` feedback packet after every
    /// `feedback_every` received packets (source or coded).
    pub(crate) feedback_every: u32,
}

impl DecoderConfig {
    pub fn builder(symbol_size: usize) -> DecoderConfigBuilder {
        DecoderConfigBuilder::new(symbol_size)
    }

    pub fn symbol_size(&self) -> usize {
        self.symbol_size
    }
    pub fn field(&self) -> Field {
        self.field
    }
    pub fn max_matrix_rows(&self) -> usize {
        self.max_matrix_rows
    }
    pub fn feedback_every(&self) -> u32 {
        self.feedback_every
    }
}

/// Builder for [`DecoderConfig`].
#[derive(Debug)]
pub struct DecoderConfigBuilder {
    symbol_size: usize,
    field: Field,
    max_matrix_rows: usize,
    feedback_every: u32,
}

impl DecoderConfigBuilder {
    fn new(symbol_size: usize) -> Self {
        Self {
            symbol_size,
            field: Field::Gf2_8,
            max_matrix_rows: 512,
            feedback_every: 16,
        }
    }

    pub fn field(mut self, f: Field) -> Self {
        self.field = f;
        self
    }
    pub fn max_matrix_rows(mut self, n: usize) -> Self {
        self.max_matrix_rows = n;
        self
    }
    pub fn feedback_every(mut self, n: u32) -> Self {
        self.feedback_every = n;
        self
    }

    pub fn build(self) -> Result<DecoderConfig> {
        if self.symbol_size == 0 || self.symbol_size > 65_535 {
            return Err(DelpError::InvalidSymbolSize(self.symbol_size));
        }
        Ok(DecoderConfig {
            symbol_size: self.symbol_size,
            field: self.field,
            max_matrix_rows: self.max_matrix_rows,
            feedback_every: self.feedback_every,
        })
    }
}

/// Derive a [`DecoderConfig`] that is guaranteed to be compatible with a
/// given [`EncoderConfig`] (same `symbol_size` and `field`).
impl From<&EncoderConfig> for DecoderConfigBuilder {
    fn from(enc: &EncoderConfig) -> Self {
        DecoderConfigBuilder::new(enc.symbol_size).field(enc.field)
    }
}
