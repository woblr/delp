// `DelpPacket` is an enum where one variant owns parsed `FeedbackPacket`
// data while others are zero-copy `&'a [u8]` views.  The size imbalance is
// intentional — flattening it would force heap allocation on the hot
// zero-copy paths.
#![allow(clippy::large_enum_variant)]

//! # delp
//!
//! **Delp** is a production-grade, pure-Rust FEC (forward error correction)
//! library for lossy networks.  It uses elastic sliding-window coding —
//! the sender XOR-combines source packets into coded (repair) packets using
//! Galois field arithmetic, and the receiver reconstructs lost packets by
//! solving a linear system over the same field.  Unlike block codes, the
//! window adapts in real time based on receiver feedback, minimising
//! per-symbol delivery latency.
//!
//! ## Quick start
//!
//! ```rust
//! use delp::{
//!     config::EncoderConfig,
//!     codec::{DefaultEncoder, DefaultDecoder, EncoderOutput, DecoderEvent},
//!     wire::{source::SourcePacket, coded::CodedPacket},
//! };
//! use bytes::Bytes;
//!
//! // 1. Configure encoder and decoder with matching symbol size
//! let enc_cfg = EncoderConfig::builder(256)
//!     .window_capacity(128)
//!     .fec_rate(1, 4)        // 25 % redundancy
//!     .build()
//!     .unwrap();
//!
//! let dec_cfg = delp::config::DecoderConfig::builder(256).build().unwrap();
//!
//! let mut enc = DefaultEncoder::with_defaults(enc_cfg);
//! let mut dec = DefaultDecoder::with_defaults(dec_cfg);
//!
//! // 2. Submit source data
//! let symbol = Bytes::from(vec![0x42u8; 256]);
//! let packets: Vec<_> = enc.submit_source(symbol).unwrap();
//!
//! // 3. Feed packets to decoder (possibly via a lossy network)
//! for pkt in packets {
//!     match pkt {
//!         EncoderOutput::Source(raw) => {
//!             let sp = SourcePacket::parse(&raw).unwrap();
//!             let events = dec.handle_source(&sp).unwrap();
//!             for ev in events {
//!                 if let DecoderEvent::SourceReady { id, data } = ev {
//!                     println!("Delivered symbol {id}: {} bytes", data.len());
//!                 }
//!             }
//!         }
//!         EncoderOutput::Coded(raw) => {
//!             let cp = CodedPacket::parse(&raw).unwrap();
//!             let _ = dec.handle_coded(&cp).unwrap();
//!         }
//!     }
//! }
//! ```
//!
//! ## Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────┐
//! │  config      — EncoderConfig / DecoderConfig       │
//! │  error       — DelpError enum                      │
//! ├────────────────────────────────────────────────────┤
//! │  gf/         — GF(2⁴) + GF(2⁸) + SIMD mul_acc    │
//! ├────────────────────────────────────────────────────┤
//! │  wire/       — binary packet wire format           │
//! │    common    — CommonHeader                        │
//! │    source    — SourcePacket                        │
//! │    coded     — CodedPacket                         │
//! │    feedback  — FeedbackPacket (window update)      │
//! │    ev/       — EncodingVector + 4 ID storage fmts  │
//! ├────────────────────────────────────────────────────┤
//! │  policy/     — Pluggable strategy traits           │
//! │    WindowPolicy / CongestionControl                │
//! │    FecRateController / FeedbackPolicy              │
//! │    defaults  — AnyAck, AllAck, Quorum, Adaptive    │
//! ├────────────────────────────────────────────────────┤
//! │  codec/      — Pure state machines (no I/O)        │
//! │    encoder   — Encoder<W,C,F> + sliding window     │
//! │    decoder   — Decoder<P> + matrix + buffer        │
//! └────────────────────────────────────────────────────┘
//! ```
//!
//! ## Design principles
//!
//! - **No I/O inside the codec** — all methods are synchronous state transitions;
//!   transport is entirely the caller's concern.
//! - **Zero-copy payloads** — symbol data is passed as [`bytes::Bytes`]; no
//!   unnecessary copies on the hot path.
//! - **SIMD-accelerated GF arithmetic** — AVX2 / SSSE3 / NEON / scalar
//!   dispatch via `std::arch`; all selected at compile time.
//! - **Pluggable strategies** — window eviction, congestion control, FEC
//!   rate, and feedback scheduling are all swappable without forking the codec.
//! - **Compile-time GF tables** — log/exp/mul tables are `const fn`; zero
//!   runtime initialisation cost.

pub mod codec;
pub mod config;
pub mod error;
pub mod gf;
pub mod policy;
pub mod wire;

#[cfg(feature = "async")]
pub mod transport;

// ── Top-level convenience re-exports ─────────────────────────────────────

pub use config::{BackpressureMode, DecoderConfig, EncoderConfig, Field, MatrixStrategy};
pub use error::{DelpError, Result};

pub use codec::{Decoder, DecoderEvent, DefaultDecoder, DefaultEncoder, Encoder, EncoderOutput};

pub use policy::{
    CodedSymbolId, CongestionControl, FecRateController, FeedbackPolicy, ReceiverAckState,
    ReceiverId, SourceSymbolId, WindowPolicy,
};

pub use policy::defaults::{
    AdaptiveFecRate, AllAckPolicy, AnyAckPolicy, ConstantFecRate, ConstantFeedbackPolicy,
    ImmediateFeedbackPolicy, NoCongestionControl, QuorumAckPolicy,
};
