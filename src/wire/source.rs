use crate::error::{Result, DelpError};
use super::common::{CommonHeader, PacketType};

/// Delp protocol §5.1 — Source Packet.
///
/// ```text
/// [CommonHeader][CCI?][TSI?][source_symbol_id: u32][payload: symbol_size bytes]
/// ```
///
/// The payload is a zero-copy `Bytes` slice referencing the original buffer.
#[derive(Debug, Clone)]
pub struct SourcePacket<'a> {
    pub header:           CommonHeader,
    /// Raw CCI bytes (0, 4, 8, or 12 bytes).
    pub cci:              &'a [u8],
    /// Transport Session Identifier (present when `header.has_tsi()` is true).
    pub tsi:              Option<u32>,
    pub source_symbol_id: u32,
    pub payload:          &'a [u8],
}

impl<'a> SourcePacket<'a> {
    // Minimum body size: 4 bytes for source_symbol_id (payload may be 0 in tests)
    const BODY_MIN: usize = 4;

    // ── Parsing ──────────────────────────────────────────────────────────

    pub fn parse(buf: &'a [u8]) -> Result<Self> {
        let header = CommonHeader::parse(buf)?;
        if header.packet_type()? != PacketType::Source {
            return Err(DelpError::UnknownPacketType(buf[3]));
        }

        let body_off = header.body_offset();
        if buf.len() < body_off + Self::BODY_MIN {
            return Err(DelpError::BufferTooShort {
                needed:    body_off + Self::BODY_MIN,
                available: buf.len(),
            });
        }

        let cci_end = CommonHeader::SIZE + header.cci_words() as usize * 4;
        let cci = &buf[CommonHeader::SIZE..cci_end];

        let (tsi, id_off) = if header.has_tsi() {
            let v = u32::from_be_bytes(buf[cci_end..cci_end + 4].try_into().unwrap());
            (Some(v), cci_end + 4)
        } else {
            (None, cci_end)
        };

        let source_symbol_id = u32::from_be_bytes(buf[id_off..id_off + 4].try_into().unwrap());
        let payload = &buf[id_off + 4..];

        Ok(Self { header, cci, tsi, source_symbol_id, payload })
    }

    // ── Serialisation ────────────────────────────────────────────────────

    /// Serialise into a newly allocated `Vec<u8>`.
    ///
    /// `cci_bytes` must be 0, 4, 8, or 12 bytes.
    pub fn serialise(
        source_symbol_id: u32,
        payload: &[u8],
        cci_bytes: &[u8],
        tsi: Option<u32>,
    ) -> Vec<u8> {
        assert!(cci_bytes.len() % 4 == 0 && cci_bytes.len() <= 12);
        let cci_words = (cci_bytes.len() / 4) as u8;
        let has_tsi   = tsi.is_some();

        // hdr_len in 32-bit words:
        //   1 (common) + cci_words + (1 if tsi) + 1 (src_id)
        let hdr_words = 1u8 + cci_words + has_tsi as u8 + 1;

        let hdr = CommonHeader::new(
            PacketType::Source,
            super::common::CciLength::from_words(cci_words),
            has_tsi,
            hdr_words,
        );

        let mut out = Vec::with_capacity(hdr_words as usize * 4 + payload.len());
        out.extend_from_slice(&hdr.to_bytes());
        out.extend_from_slice(cci_bytes);
        if let Some(t) = tsi {
            out.extend_from_slice(&t.to_be_bytes());
        }
        out.extend_from_slice(&source_symbol_id.to_be_bytes());
        out.extend_from_slice(payload);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_no_cci_no_tsi() {
        let payload = b"hello delp world!";
        let raw = SourcePacket::serialise(42, payload, &[], None);
        let pkt = SourcePacket::parse(&raw).unwrap();
        assert_eq!(pkt.source_symbol_id, 42);
        assert_eq!(pkt.payload, payload);
        assert_eq!(pkt.tsi, None);
        assert_eq!(pkt.cci.len(), 0);
    }

    #[test]
    fn round_trip_with_tsi() {
        let payload = vec![0xABu8; 32];
        let raw = SourcePacket::serialise(0xDEAD_BEEF, &payload, &[], Some(99));
        let pkt = SourcePacket::parse(&raw).unwrap();
        assert_eq!(pkt.source_symbol_id, 0xDEAD_BEEF);
        assert_eq!(pkt.tsi, Some(99));
        assert_eq!(pkt.payload, payload.as_slice());
    }

    #[test]
    fn round_trip_with_cci() {
        let cci = [0x01u8, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]; // 2 words
        let payload = b"data";
        let raw = SourcePacket::serialise(7, payload, &cci, None);
        let pkt = SourcePacket::parse(&raw).unwrap();
        assert_eq!(pkt.cci, &cci);
        assert_eq!(pkt.payload, payload);
    }
}