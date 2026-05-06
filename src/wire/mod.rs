pub mod coded;
pub mod common;
pub mod ev;
pub mod feedback;
pub mod source;

pub use coded::CodedPacket;
pub use common::{CciLength, CommonHeader, PacketType};
pub use ev::EncodingVector;
pub use feedback::FeedbackPacket;
pub use source::SourcePacket;

use crate::error::{DelpError, Result};

/// Top-level Delp packet — parsed from raw bytes without copying the payload.
#[derive(Debug)]
pub enum DelpPacket<'a> {
    Source(SourcePacket<'a>),
    Coded(CodedPacket<'a>),
    Feedback(FeedbackPacket),
}

impl<'a> DelpPacket<'a> {
    /// Parse a Delp packet from a raw byte slice.
    ///
    /// The slice must contain exactly one packet; trailing bytes are an error.
    /// No data is copied — payload references point into `buf`.
    pub fn parse(buf: &'a [u8]) -> Result<Self> {
        if buf.len() < CommonHeader::SIZE {
            return Err(DelpError::BufferTooShort {
                needed: CommonHeader::SIZE,
                available: buf.len(),
            });
        }

        let hdr = CommonHeader::parse(buf)?;

        match hdr.packet_type()? {
            PacketType::Source => Ok(DelpPacket::Source(SourcePacket::parse(buf)?)),
            PacketType::Coded => Ok(DelpPacket::Coded(CodedPacket::parse(buf)?)),
            PacketType::Feedback => Ok(DelpPacket::Feedback(FeedbackPacket::parse(buf)?)),
        }
    }
}
