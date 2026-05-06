//! Async UDP transport layer for the Delp FEC codec.
//!
//! Enabled via the `async` Cargo feature:
//!
//! ```toml
//! [dependencies]
//! delp = { version = "1", features = ["async"] }
//! ```
//!
//! # Architecture
//!
//! The codec is a pure synchronous state machine; this module adds
//! the I/O layer on top using `tokio::net::UdpSocket`.
//!
//! ```text
//! Application ──Bytes──► FecSender ──UDP──► FecReceiver ──(id, Bytes)──► Application
//!                            ▲                  │
//!                            └──── feedback ────┘
//! ```
//!
//! For bidirectional communication over a single socket, use [`DelpSession`].
//!
//! # Quick start
//!
//! See `examples/udp_fec.rs` for a complete working example.

pub mod receiver;
pub mod sender;
pub mod session;

pub use receiver::FecReceiver;
pub use sender::FecSender;
pub use session::DelpSession;

// ── Transport error ───────────────────────────────────────────────────────

use crate::error::DelpError;

/// Combined error type for transport operations.
///
/// Wraps both codec-level errors ([`DelpError`]) and OS-level I/O errors.
#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    /// Codec error (wire format, field arithmetic, window management).
    #[error("codec error: {0}")]
    Codec(#[from] DelpError),

    /// OS-level socket / I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

pub type TransportResult<T> = Result<T, TransportError>;
