pub mod coef_gen;
pub mod id_storage;

use crate::config::Field;
use crate::error::{DelpError, Result};
use id_storage::IdStorageFormat;
use smallvec::SmallVec;

/// Delp Encoding Vector.
///
/// ```text
/// Byte 0: EV_LEN (8 bits) — length in 32-bit words
/// Byte 1: [CCGI:4][I:2][C:1][G:1]
/// Byte 2: NB_IDS (8 bits)
/// Byte 3: NB_COEFS (8 bits)
/// Byte 4..7: FIRST_SOURCE_ID (32 bits)
/// [If G=1] Byte 8: GENERATION (8 bits); bytes 9..11: padding (zero)
/// Then: source ID bit-field (I-format dependent, padded to 32-bit)
/// Then: coefficient bit-field (if C=1, else implicit Vandermonde)
/// ```
///
/// **`G` (Generation Present):** delp extension over RFC 9407.  Repurposes the
/// reserved `V` bit to signal a 1-byte `GENERATION` field after `FIRST_SOURCE_ID`.
/// Cauchy uses generation rotation to support unlimited-length sessions; the
/// receiver rotates the y-point assignment by `(coded_id + gen * K) mod cycle`
/// to keep successive cycles linearly independent.
///
/// `G=0` packets are bit-for-bit identical to RFC 9407 EVs (full backward
/// compatibility for sources where generation never advances).
///
/// `CCGI` selects GF(2⁴)=0 or GF(2⁸)=1.
#[derive(Debug, Clone)]
pub struct EncodingVector {
    pub field: Field,
    pub coded_id: u32,
    /// Generation counter — when non-zero, serialised as the optional
    /// `GENERATION` byte and the `G` flag is set.  Used by Cauchy to rotate
    /// the y-point set so wrapped coded IDs produce independent equations.
    pub generation: u8,
    pub id_format: IdStorageFormat,
    pub source_ids: SmallVec<[u32; 64]>,
    /// Explicit per-source coefficients (empty when C=0 → Vandermonde).
    pub coefficients: SmallVec<[u8; 64]>,
}

impl EncodingVector {
    // ── Constructors ─────────────────────────────────────────────────────

    /// Build an encoding vector with implicit Vandermonde coefficients
    /// (most bandwidth-efficient; C=0 on the wire).  Generation defaults to 0
    /// — Vandermonde does not use generation rotation.
    pub fn vandermonde(field: Field, coded_id: u32, source_ids: SmallVec<[u32; 64]>) -> Self {
        let id_format = IdStorageFormat::best_for(&source_ids);
        Self {
            field,
            coded_id,
            generation: 0,
            id_format,
            source_ids,
            coefficients: SmallVec::new(),
        }
    }

    /// Build an encoding vector with explicit per-source coefficients (C=1).
    /// Generation defaults to 0; use [`Self::with_generation`] to set it for
    /// Cauchy long-session rotation.
    pub fn explicit(
        field: Field,
        coded_id: u32,
        source_ids: SmallVec<[u32; 64]>,
        coefficients: SmallVec<[u8; 64]>,
    ) -> Self {
        assert_eq!(source_ids.len(), coefficients.len());
        let id_format = IdStorageFormat::best_for(&source_ids);
        Self {
            field,
            coded_id,
            generation: 0,
            id_format,
            source_ids,
            coefficients,
        }
    }

    /// Set the generation counter (for Cauchy long-session y-point rotation).
    pub fn with_generation(mut self, generation: u8) -> Self {
        self.generation = generation;
        self
    }

    pub fn has_explicit_coefs(&self) -> bool {
        !self.coefficients.is_empty()
    }
    pub fn has_generation(&self) -> bool {
        self.generation != 0
    }
    pub fn nb_ids(&self) -> u8 {
        self.source_ids.len() as u8
    }
    pub fn nb_coefs(&self) -> u8 {
        self.coefficients.len() as u8
    }

    // ── Serialisation ────────────────────────────────────────────────────

    pub fn serialise(&self) -> Vec<u8> {
        let mut id_bits = id_storage::encode(&self.source_ids, self.id_format);
        let id_padded = pad32(id_bits.len());
        id_bits.resize(id_padded, 0);

        let coef_bytes = if self.has_explicit_coefs() {
            let bits = match self.field {
                Field::Gf2_8 => self.coefficients.to_vec(),
                Field::Gf2_4 => pack_nibbles(&self.coefficients),
            };
            let padded = pad32(bits.len());
            let mut v = bits;
            v.resize(padded, 0);
            v
        } else {
            Vec::new()
        };

        let has_gen = self.has_generation();
        let gen_block = if has_gen { 4 } else { 0 }; // 1 byte + 3 padding
                                                     // Fixed part: 8 bytes (ev_len + flags + nb_ids + nb_coefs + first_src_id)
                                                     //           + optional 4 bytes for the generation block
        let body_len = 8 + gen_block + id_padded + coef_bytes.len();
        let ev_len_words = body_len.div_ceil(4) as u8;

        let ccgi_bit: u8 = match self.field {
            Field::Gf2_4 => 0,
            Field::Gf2_8 => 1,
        };
        let i_bits: u8 = self.id_format.wire_value();
        let c_bit: u8 = self.has_explicit_coefs() as u8;
        let g_bit: u8 = has_gen as u8;
        // bit layout: [CCGI:4][I:2][C:1][G:1]
        let byte1 = (ccgi_bit << 4) | (i_bits << 2) | (c_bit << 1) | g_bit;

        let first_src_id = self.source_ids.first().copied().unwrap_or(0);

        let mut out = Vec::with_capacity(body_len + 4);
        out.push(ev_len_words);
        out.push(byte1);
        out.push(self.nb_ids());
        out.push(self.nb_coefs());
        out.extend_from_slice(&first_src_id.to_be_bytes());
        if has_gen {
            out.push(self.generation);
            out.extend_from_slice(&[0u8; 3]); // pad to 32-bit boundary
        }
        out.extend_from_slice(&id_bits);
        out.extend_from_slice(&coef_bytes);
        out
    }

    // ── Parsing ──────────────────────────────────────────────────────────

    /// Parse an encoding vector from `buf` starting at offset 0.
    /// Returns the EV and the number of bytes consumed.
    pub fn parse(buf: &[u8], coded_id: u32) -> Result<(Self, usize)> {
        if buf.len() < 8 {
            return Err(DelpError::BufferTooShort {
                needed: 8,
                available: buf.len(),
            });
        }

        let ev_len_words = buf[0] as usize;
        let ev_bytes = ev_len_words * 4;
        if buf.len() < ev_bytes {
            return Err(DelpError::EncodingVectorOverflow {
                ev_len_words: buf[0],
            });
        }

        let byte1 = buf[1];
        let ccgi_bit = (byte1 >> 4) & 0xF;
        let i_bits = (byte1 >> 2) & 0x3;
        let c_bit = (byte1 >> 1) & 0x1;
        let g_bit = byte1 & 0x1;

        let field = match ccgi_bit {
            0 => Field::Gf2_4,
            1 => Field::Gf2_8,
            _ => {
                return Err(DelpError::MalformedEncodingVector {
                    reason: "invalid CCGI",
                })
            }
        };

        let nb_ids = buf[2] as usize;
        let nb_coefs = buf[3] as usize;

        if c_bit == 0 && nb_coefs != 0 {
            return Err(DelpError::CoefficientFieldInconsistency);
        }
        if c_bit == 1 && nb_coefs != nb_ids {
            return Err(DelpError::CoefficientFieldInconsistency);
        }

        let first_src_id = u32::from_be_bytes(buf[4..8].try_into().unwrap());

        // Optional GENERATION block: 1 byte + 3 padding when G=1.
        let (generation, body_off) = if g_bit == 1 {
            if buf.len() < 12 {
                return Err(DelpError::BufferTooShort {
                    needed: 12,
                    available: buf.len(),
                });
            }
            (buf[8], 12)
        } else {
            (0u8, 8)
        };

        let id_format = IdStorageFormat::from_wire(i_bits)?;
        let (source_ids, id_bytes_consumed) =
            id_storage::decode(&buf[body_off..ev_bytes], nb_ids, first_src_id, id_format)?;

        let coef_offset = body_off + pad32(id_bytes_consumed);
        let coefficients = if c_bit == 1 {
            let coef_bytes_needed = match field {
                Field::Gf2_8 => nb_coefs,
                Field::Gf2_4 => nb_coefs.div_ceil(2),
            };
            if buf.len() < coef_offset + coef_bytes_needed {
                return Err(DelpError::BufferTooShort {
                    needed: coef_offset + coef_bytes_needed,
                    available: buf.len(),
                });
            }
            let raw = &buf[coef_offset..coef_offset + coef_bytes_needed];
            match field {
                Field::Gf2_8 => SmallVec::from_slice(raw),
                Field::Gf2_4 => unpack_nibbles(raw, nb_coefs),
            }
        } else {
            SmallVec::new()
        };

        Ok((
            Self {
                field,
                coded_id,
                generation,
                id_format,
                source_ids: SmallVec::from(source_ids),
                coefficients,
            },
            ev_bytes,
        ))
    }
}

// ── Nibble packing for GF(2^4) coefficients ──────────────────────────────

fn pack_nibbles(coefs: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(coefs.len().div_ceil(2));
    for chunk in coefs.chunks(2) {
        let hi = chunk[0] & 0x0F;
        let lo = chunk.get(1).copied().unwrap_or(0) & 0x0F;
        out.push((hi << 4) | lo);
    }
    out
}

fn unpack_nibbles(bytes: &[u8], count: usize) -> SmallVec<[u8; 64]> {
    let mut out = SmallVec::with_capacity(count);
    for byte in bytes {
        out.push((byte >> 4) & 0x0F);
        if out.len() < count {
            out.push(byte & 0x0F);
        }
    }
    out.truncate(count);
    out
}

/// Round up to next 32-bit boundary.
#[inline(always)]
fn pad32(len: usize) -> usize {
    (len + 3) & !3
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contiguous_ids(start: u32, count: u32) -> SmallVec<[u32; 64]> {
        (start..start + count).collect()
    }

    #[test]
    fn vandermonde_round_trip_gf2_8() {
        let ids = contiguous_ids(10, 8);
        let ev = EncodingVector::vandermonde(Field::Gf2_8, 3, ids.clone());
        let bytes = ev.serialise();
        let (parsed, consumed) = EncodingVector::parse(&bytes, 3).unwrap();
        assert_eq!(consumed, bytes.len());
        assert_eq!(parsed.source_ids.as_slice(), ids.as_slice());
        assert!(!parsed.has_explicit_coefs());
        assert_eq!(parsed.field, Field::Gf2_8);
    }

    #[test]
    fn explicit_coefs_round_trip_gf2_8() {
        let ids: SmallVec<[u32; 64]> = (0..4).collect();
        let coefs: SmallVec<[u8; 64]> = smallvec::smallvec![0x01, 0x02, 0x04, 0x08];
        let ev = EncodingVector::explicit(Field::Gf2_8, 1, ids.clone(), coefs.clone());
        let bytes = ev.serialise();
        let (parsed, _) = EncodingVector::parse(&bytes, 1).unwrap();
        assert_eq!(parsed.source_ids.as_slice(), ids.as_slice());
        assert_eq!(parsed.coefficients.as_slice(), coefs.as_slice());
    }

    #[test]
    fn nibble_pack_unpack_roundtrip() {
        let coefs: Vec<u8> = (1..=6).collect();
        let packed = pack_nibbles(&coefs);
        let unpacked = unpack_nibbles(&packed, coefs.len());
        assert_eq!(unpacked.as_slice(), coefs.as_slice());
    }
}
