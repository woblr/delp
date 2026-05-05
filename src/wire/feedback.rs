use crate::error::{Result, DelpError};
use super::common::{CommonHeader, PacketType, CciLength};

/// Delp protocol §5.3 — Window Update (Feedback) Packet.
///
/// ```text
/// [CommonHeader][CCI?][TSI?]
/// [nb_missing_src: u32][nb_not_used_coded: u32]
/// [first_src_id: u32][plr: u8][sack_size: u8][pad: u16]
/// [sack_vector: sack_size bytes, padded to 32-bit boundary]
/// ```
///
/// The SACK vector is MSB-first: bit `i` of byte `i/8` corresponds to
/// source symbol `first_src_id + i`.  A set bit means "acknowledged".
#[derive(Debug, Clone)]
pub struct FeedbackPacket {
    pub tsi:              Option<u32>,
    pub nb_missing_src:   u32,
    pub nb_not_used_coded: u32,
    /// First source ID in the cumulative ACK / SACK window.
    pub first_src_id:     u32,
    /// Packet loss rate, encoded as `plr * 256 / 100` (saturating at 255).
    pub plr_raw:          u8,
    /// SACK bit vector — each bit acknowledges one source symbol ID.
    pub sack:             Vec<u8>,
}

impl FeedbackPacket {
    // ── PLR helpers ──────────────────────────────────────────────────────

    /// Encode a loss rate in \[0.0, 1.0\] to the wire `plr` byte.
    pub fn encode_plr(loss: f64) -> u8 {
        (loss.clamp(0.0, 1.0) * 256.0).min(255.0) as u8
    }

    /// Decode the wire `plr` byte to a loss rate in \[0.0, 1.0\].
    pub fn decode_plr(raw: u8) -> f64 { raw as f64 / 256.0 }

    // ── Build a feedback packet from a received-ACK bit-set ──────────────

    /// Build a feedback packet.
    ///
    /// `acked_ids` — sorted slice of source IDs that have been received
    /// and should be acknowledged.  The smallest ID becomes `first_src_id`.
    pub fn build(
        first_src_id:      u32,
        nb_missing_src:    u32,
        nb_not_used_coded: u32,
        loss_rate:         f64,
        acked_ids:         &[u32],
    ) -> Self {
        let sack = build_sack(first_src_id, acked_ids);
        Self {
            tsi: None,
            nb_missing_src,
            nb_not_used_coded,
            first_src_id,
            plr_raw: Self::encode_plr(loss_rate),
            sack,
        }
    }

    // ── Serialisation ────────────────────────────────────────────────────

    pub fn serialise(&self) -> Vec<u8> {
        let sack_bytes      = self.sack.len();
        let sack_padded     = (sack_bytes + 3) & !3; // round up to 32-bit
        let tsi_words       = self.tsi.is_some() as u8;
        // hdr: 1(common) + tsi_words + 2(nb_missing+nb_not_used) + 1(first+plr+sack_size+pad)
        let hdr_words: u8   = 1 + tsi_words + 2 + 1;
        let total           = hdr_words as usize * 4 + sack_padded;

        let hdr = CommonHeader::new(
            PacketType::Feedback,
            CciLength::Zero,
            self.tsi.is_some(),
            hdr_words,
        );

        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(&hdr.to_bytes());
        if let Some(t) = self.tsi {
            out.extend_from_slice(&t.to_be_bytes());
        }
        out.extend_from_slice(&self.nb_missing_src.to_be_bytes());
        out.extend_from_slice(&self.nb_not_used_coded.to_be_bytes());
        out.extend_from_slice(&self.first_src_id.to_be_bytes());
        out.push(self.plr_raw);
        out.push(sack_bytes as u8);
        out.extend_from_slice(&[0u8; 2]); // padding
        out.extend_from_slice(&self.sack);
        // Pad SACK to 32-bit boundary
        for _ in sack_bytes..sack_padded {
            out.push(0);
        }
        out
    }

    // ── Parsing ──────────────────────────────────────────────────────────

    pub fn parse(buf: &[u8]) -> Result<Self> {
        let header  = CommonHeader::parse(buf)?;
        let off     = header.body_offset();

        // Fixed body: nb_missing(4) + nb_not_used(4) + first_src_id(4)
        //           + plr(1) + sack_size(1) + pad(2) = 16 bytes
        if buf.len() < off + 16 {
            return Err(DelpError::BufferTooShort { needed: off + 16, available: buf.len() });
        }

        let tsi = if header.has_tsi() {
            let t = u32::from_be_bytes(buf[CommonHeader::SIZE..CommonHeader::SIZE + 4].try_into().unwrap());
            Some(t)
        } else { None };

        let nb_missing_src   = u32::from_be_bytes(buf[off..off + 4].try_into().unwrap());
        let nb_not_used_coded = u32::from_be_bytes(buf[off + 4..off + 8].try_into().unwrap());
        let first_src_id     = u32::from_be_bytes(buf[off + 8..off + 12].try_into().unwrap());
        let plr_raw          = buf[off + 12];
        let sack_size        = buf[off + 13] as usize;
        // skip 2 padding bytes → off + 16

        let sack_start = off + 16;
        if buf.len() < sack_start + sack_size {
            return Err(DelpError::BufferTooShort {
                needed:    sack_start + sack_size,
                available: buf.len(),
            });
        }
        let sack = buf[sack_start..sack_start + sack_size].to_vec();

        Ok(Self { tsi, nb_missing_src, nb_not_used_coded, first_src_id, plr_raw, sack })
    }

    // ── SACK query ───────────────────────────────────────────────────────

    /// Returns `true` if source symbol `id` is acknowledged in this packet.
    pub fn is_acked(&self, id: u32) -> bool {
        if id < self.first_src_id { return false; }
        let offset = (id - self.first_src_id) as usize;
        let byte   = offset / 8;
        let bit    = 7 - (offset % 8); // MSB-first
        if byte >= self.sack.len() { return false; }
        (self.sack[byte] >> bit) & 1 == 1
    }

    /// Iterator over all acknowledged source symbol IDs.
    pub fn acked_ids(&self) -> impl Iterator<Item = u32> + '_ {
        let base = self.first_src_id;
        self.sack.iter().enumerate().flat_map(move |(byte_idx, &byte)| {
            (0..8u32).filter_map(move |bit| {
                if (byte >> (7 - bit)) & 1 == 1 {
                    Some(base + byte_idx as u32 * 8 + bit)
                } else {
                    None
                }
            })
        })
    }
}

// ── Internal: build SACK bit vector ─────────────────────────────────────

fn build_sack(first_src_id: u32, acked_ids: &[u32]) -> Vec<u8> {
    if acked_ids.is_empty() { return Vec::new(); }

    let last_id   = *acked_ids.iter().max().unwrap();
    if last_id < first_src_id { return Vec::new(); }

    let span      = (last_id - first_src_id) as usize + 1;
    let byte_len  = (span + 7) / 8;
    let mut sack  = vec![0u8; byte_len];

    for &id in acked_ids {
        if id < first_src_id { continue; }
        let offset = (id - first_src_id) as usize;
        let byte   = offset / 8;
        let bit    = 7 - (offset % 8);
        sack[byte] |= 1 << bit;
    }
    sack
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sack_round_trip() {
        let acked: Vec<u32> = vec![10, 11, 13, 15];
        let pkt = FeedbackPacket::build(10, 2, 0, 0.1, &acked);
        assert!( pkt.is_acked(10));
        assert!( pkt.is_acked(11));
        assert!(!pkt.is_acked(12));
        assert!( pkt.is_acked(13));
        assert!(!pkt.is_acked(14));
        assert!( pkt.is_acked(15));

        let collected: Vec<u32> = pkt.acked_ids().collect();
        assert_eq!(collected, vec![10, 11, 13, 15]);
    }

    #[test]
    fn serialise_parse_round_trip() {
        let acked: Vec<u32> = (0..24).filter(|i| i % 3 != 0).collect();
        let original = FeedbackPacket::build(0, 5, 2, 0.05, &acked);
        let bytes    = original.serialise();
        let parsed   = FeedbackPacket::parse(&bytes).unwrap();

        assert_eq!(parsed.first_src_id,      original.first_src_id);
        assert_eq!(parsed.nb_missing_src,     original.nb_missing_src);
        assert_eq!(parsed.nb_not_used_coded,  original.nb_not_used_coded);
        assert_eq!(parsed.plr_raw,            original.plr_raw);
        assert_eq!(parsed.sack,               original.sack);
    }

    #[test]
    fn plr_encoding() {
        assert_eq!(FeedbackPacket::encode_plr(0.0),  0);
        assert_eq!(FeedbackPacket::encode_plr(1.0),  255);
        let v = FeedbackPacket::encode_plr(0.1);
        let r = FeedbackPacket::decode_plr(v);
        assert!((r - 0.1).abs() < 0.01);
    }
}