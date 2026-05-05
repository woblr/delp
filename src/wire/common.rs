use zerocopy::{FromBytes, IntoBytes, KnownLayout, Immutable};
use crate::error::{Result, DelpError};

/// Binary common header — 4 bytes, present in every packet type.
///
/// ```text
/// Byte 0: [V:4][C:2][S:1][Rsvd:1]
/// Byte 1: [Rsvd:8]
/// Byte 2: [HDR_LEN:8]   — total header length in 32-bit words
/// Byte 3: [PKT_TYPE:8]
/// ```
///
/// `zerocopy::FromBytes` lets us safely reinterpret a `&[u8]` as a
/// `&CommonHeader` with zero allocation and no manual byte indexing.
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable)]
#[repr(C)]
pub struct CommonHeader {
    pub(crate) flags:        u8,  // [V:4][C:2][S:1][Rsvd:1]
    pub(crate) reserved:     u8,
    pub(crate) hdr_len_words: u8, // total header length in 32-bit words
    pub(crate) pkt_type:     u8,
}

impl CommonHeader {
    pub const SIZE: usize  = 4;
    pub const VERSION: u8  = 1;

    // ── Construction ────────────────────────────────────────────────────

    pub fn new(
        pkt_type:      PacketType,
        cci:           CciLength,
        has_tsi:       bool,
        hdr_len_words: u8,
    ) -> Self {
        let flags = (Self::VERSION << 4)
            | ((cci as u8) << 2)
            | ((has_tsi as u8) << 1);
        Self { flags, reserved: 0, hdr_len_words, pkt_type: pkt_type as u8 }
    }

    #[inline]
    pub fn to_bytes(self) -> [u8; 4] {
        [self.flags, self.reserved, self.hdr_len_words, self.pkt_type]
    }

    // ── Zero-copy parsing ────────────────────────────────────────────────

    /// Parse the common header from the start of `buf` using `zerocopy`.
    ///
    /// No data is copied — the returned `CommonHeader` is a value copy of
    /// the 4 header bytes (4-byte struct, always cheap to copy).
    pub fn parse(buf: &[u8]) -> Result<Self> {
        if buf.len() < Self::SIZE {
            return Err(DelpError::BufferTooShort {
                needed:    Self::SIZE,
                available: buf.len(),
            });
        }

        // zerocopy: safe cast — CommonHeader is FromBytes + repr(C)
        let hdr = *Self::ref_from_bytes(&buf[..Self::SIZE])
            .map_err(|_| DelpError::BufferTooShort {
                needed:    Self::SIZE,
                available: buf.len(),
            })?;

        let ver = hdr.version();
        if ver != Self::VERSION {
            return Err(DelpError::UnsupportedVersion(ver));
        }

        let hdr_len_bytes = hdr.hdr_len_words as usize * 4;
        if hdr_len_bytes > buf.len() {
            return Err(DelpError::InvalidHeaderLength {
                hdr_len_words: hdr.hdr_len_words,
                packet_len:    buf.len(),
            });
        }

        Ok(hdr)
    }

    // ── Field accessors ──────────────────────────────────────────────────

    #[inline] pub fn version(self) -> u8         { self.flags >> 4 }
    #[inline] pub fn cci_words(self) -> u8        { (self.flags >> 2) & 0x3 }
    #[inline] pub fn has_tsi(self) -> bool        { (self.flags >> 1) & 0x1 == 1 }
    #[inline] pub fn hdr_len_bytes(self) -> usize { self.hdr_len_words as usize * 4 }

    pub fn packet_type(self) -> Result<PacketType> {
        PacketType::from_u8(self.pkt_type)
    }

    pub fn cci_length(self) -> CciLength {
        CciLength::from_words(self.cci_words())
    }

    /// Byte offset of the type-specific body (after CCI + optional TSI).
    #[inline]
    pub fn body_offset(self) -> usize {
        Self::SIZE
            + self.cci_words() as usize * 4
            + if self.has_tsi() { 4 } else { 0 }
    }
}

// ── PacketType ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PacketType {
    Source   = 0x00,
    Coded    = 0x01,
    Feedback = 0x02,
}

impl PacketType {
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            0x00 => Ok(Self::Source),
            0x01 => Ok(Self::Coded),
            0x02 => Ok(Self::Feedback),
            _    => Err(DelpError::UnknownPacketType(v)),
        }
    }
}

// ── CciLength ────────────────────────────────────────────────────────────

/// Number of 32-bit CCI words immediately after the 4-byte common header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum CciLength {
    #[default]
    Zero  = 0,
    One   = 1,
    Two   = 2,
    Three = 3,
}

impl CciLength {
    pub fn from_words(w: u8) -> Self {
        match w & 0x3 {
            0 => Self::Zero,
            1 => Self::One,
            2 => Self::Two,
            _ => Self::Three,
        }
    }
    pub fn words(self) -> u8  { self as u8 }
    pub fn bytes(self) -> usize { self as usize * 4 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_source() {
        let hdr = CommonHeader::new(PacketType::Source, CciLength::Zero, false, 1);
        let bytes = hdr.to_bytes();
        let parsed = CommonHeader::parse(&bytes).unwrap();
        assert_eq!(parsed.version(),              1);
        assert_eq!(parsed.packet_type().unwrap(), PacketType::Source);
        assert_eq!(parsed.cci_words(),            0);
        assert_eq!(parsed.has_tsi(),              false);
        assert_eq!(parsed.hdr_len_words,          1);
    }

    #[test]
    fn round_trip_with_cci_and_tsi() {
        let hdr = CommonHeader::new(PacketType::Coded, CciLength::Two, true, 5);
        let bytes = hdr.to_bytes();
        let mut buf = bytes.to_vec();
        buf.extend_from_slice(&[0u8; 20]); // enough for hdr_len=5
        let parsed = CommonHeader::parse(&buf).unwrap();
        assert_eq!(parsed.cci_words(), 2);
        assert!(parsed.has_tsi());
        // body_offset: 4 (common) + 8 (2 CCI words) + 4 (TSI) = 16
        assert_eq!(parsed.body_offset(), 16);
    }

    #[test]
    fn rejects_wrong_version() {
        let mut bytes = CommonHeader::new(PacketType::Source, CciLength::Zero, false, 1).to_bytes();
        bytes[0] = 0x20; // version 2
        assert!(matches!(
            CommonHeader::parse(&bytes),
            Err(DelpError::UnsupportedVersion(2))
        ));
    }

    #[test]
    fn rejects_short_buf() {
        assert!(matches!(
            CommonHeader::parse(&[0x10, 0x00]),
            Err(DelpError::BufferTooShort { .. })
        ));
    }

    #[test]
    fn zerocopy_ref_from_bytes() {
        // Verify zerocopy parsing gives same result as manual construction
        let hdr  = CommonHeader::new(PacketType::Feedback, CciLength::One, false, 2);
        let raw  = hdr.to_bytes();
        let hdr2 = CommonHeader::ref_from_bytes(&raw).unwrap();
        assert_eq!(hdr2.version(), hdr.version());
        assert_eq!(hdr2.cci_words(), hdr.cci_words());
    }
}