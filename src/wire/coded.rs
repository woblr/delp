use crate::error::{Result, DelpError};
use super::common::{CommonHeader, PacketType, CciLength};
use super::ev::EncodingVector;

/// Delp protocol §5.2 — Coded Packet.
///
/// ```text
/// [CommonHeader][CCI?][TSI?]
/// [coded_symbol_id: u32]
/// [EncodingVector]
/// [coded payload: symbol_size bytes]
/// ```
///
/// The payload is a zero-copy slice referencing the original buffer.
#[derive(Debug, Clone)]
pub struct CodedPacket<'a> {
    pub header:          CommonHeader,
    pub cci:             &'a [u8],
    pub tsi:             Option<u32>,
    pub coded_symbol_id: u32,
    pub ev:              EncodingVector,
    pub payload:         &'a [u8],
}

impl<'a> CodedPacket<'a> {
    const CODED_ID_SIZE: usize = 4;

    // ── Parsing ──────────────────────────────────────────────────────────

    pub fn parse(buf: &'a [u8]) -> Result<Self> {
        let header = CommonHeader::parse(buf)?;
        if header.packet_type()? != PacketType::Coded {
            return Err(DelpError::UnknownPacketType(buf[3]));
        }

        let cci_start = CommonHeader::SIZE;
        let cci_end   = cci_start + header.cci_words() as usize * 4;
        let cci       = &buf[cci_start..cci_end];

        let (tsi, after_tsi) = if header.has_tsi() {
            let v = u32::from_be_bytes(buf[cci_end..cci_end + 4].try_into().unwrap());
            (Some(v), cci_end + 4)
        } else {
            (None, cci_end)
        };

        if buf.len() < after_tsi + Self::CODED_ID_SIZE {
            return Err(DelpError::BufferTooShort {
                needed:    after_tsi + Self::CODED_ID_SIZE,
                available: buf.len(),
            });
        }

        let coded_symbol_id = u32::from_be_bytes(
            buf[after_tsi..after_tsi + 4].try_into().unwrap(),
        );
        let ev_start = after_tsi + 4;

        if buf.len() <= ev_start {
            return Err(DelpError::BufferTooShort {
                needed:    ev_start + 1,
                available: buf.len(),
            });
        }

        let (ev, ev_consumed) = EncodingVector::parse(&buf[ev_start..], coded_symbol_id)?;

        let payload_start = ev_start + ev_consumed;
        let payload       = &buf[payload_start..];

        Ok(Self { header, cci, tsi, coded_symbol_id, ev, payload })
    }

    // ── Serialisation ────────────────────────────────────────────────────

    /// Serialise a coded packet into a `Vec<u8>`.
    pub fn serialise(
        coded_symbol_id: u32,
        ev:              &EncodingVector,
        payload:         &[u8],
        cci_bytes:       &[u8],
        tsi:             Option<u32>,
    ) -> Vec<u8> {
        assert!(cci_bytes.len() % 4 == 0 && cci_bytes.len() <= 12);
        let cci_words  = (cci_bytes.len() / 4) as u8;
        let has_tsi    = tsi.is_some();

        let ev_bytes   = ev.serialise();
        let ev_words   = ((ev_bytes.len() + 3) / 4) as u8;

        // hdr_len in 32-bit words:
        //   1 (common) + cci_words + (1 if tsi) + 1 (coded_id) + ev_words
        let hdr_words  = 1u8 + cci_words + has_tsi as u8 + 1 + ev_words;

        let hdr = CommonHeader::new(
            PacketType::Coded,
            CciLength::from_words(cci_words),
            has_tsi,
            hdr_words,
        );

        let capacity = hdr_words as usize * 4 + payload.len();
        let mut out  = Vec::with_capacity(capacity);
        out.extend_from_slice(&hdr.to_bytes());
        out.extend_from_slice(cci_bytes);
        if let Some(t) = tsi { out.extend_from_slice(&t.to_be_bytes()); }
        out.extend_from_slice(&coded_symbol_id.to_be_bytes());
        out.extend_from_slice(&ev_bytes);
        // Pad EV to 32-bit boundary
        let ev_pad = (4 - ev_bytes.len() % 4) % 4;
        out.extend(core::iter::repeat(0).take(ev_pad));
        out.extend_from_slice(payload);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Field;
    use smallvec::SmallVec;

    fn make_ev(source_ids: &[u32], coded_id: u32) -> EncodingVector {
        let ids: SmallVec<[u32; 64]> = source_ids.iter().copied().collect();
        EncodingVector::vandermonde(Field::Gf2_8, coded_id, ids)
    }

    #[test]
    fn round_trip_simple() {
        let ev      = make_ev(&[0, 1, 2, 3], 1);
        let payload = vec![0xAAu8; 128];
        let raw     = CodedPacket::serialise(1, &ev, &payload, &[], None);
        let pkt     = CodedPacket::parse(&raw).unwrap();
        assert_eq!(pkt.coded_symbol_id, 1);
        assert_eq!(pkt.payload, payload.as_slice());
        assert_eq!(pkt.ev.source_ids.as_slice(), &[0, 1, 2, 3]);
    }

    #[test]
    fn round_trip_with_tsi_and_cci() {
        let ev      = make_ev(&(10..18).collect::<Vec<_>>(), 5);
        let payload = vec![0xBBu8; 64];
        let cci     = [0x01u8; 4];
        let raw     = CodedPacket::serialise(5, &ev, &payload, &cci, Some(42));
        let pkt     = CodedPacket::parse(&raw).unwrap();
        assert_eq!(pkt.tsi, Some(42));
        assert_eq!(pkt.cci, &cci);
        assert_eq!(pkt.coded_symbol_id, 5);
        assert_eq!(pkt.payload.len(), 64);
    }

    #[test]
    fn non_contiguous_source_ids() {
        let ids = vec![1u32, 3, 5, 7];
        let ev  = make_ev(&ids, 2);
        let raw = CodedPacket::serialise(2, &ev, &[0u8; 32], &[], None);
        let pkt = CodedPacket::parse(&raw).unwrap();
        assert_eq!(pkt.ev.source_ids.as_slice(), ids.as_slice());
    }
}