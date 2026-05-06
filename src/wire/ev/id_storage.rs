/// Delp protocol §5.2.1 — I-field: four source-ID storage formats.
///
/// | Wire value | Mode               | Description                                      |
/// |:----------:|:------------------:|:------------------------------------------------:|
/// | 0b00       | None               | Contiguous range `[first_src_id .. +nb_ids)`     |
/// | 0b01       | UncompressedEdges  | Raw `b_id`-bit block-edge values                 |
/// | 0b10       | CompressedList     | Delta-encoded list of every source ID            |
/// | 0b11       | CompressedEdges    | Delta-encoded block-edge boundaries              |
use crate::error::{DelpError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdStorageFormat {
    None,
    UncompressedEdges,
    CompressedList,
    CompressedEdges,
}

impl IdStorageFormat {
    pub fn from_wire(v: u8) -> Result<Self> {
        match v & 0x3 {
            0 => Ok(Self::None),
            1 => Ok(Self::UncompressedEdges),
            2 => Ok(Self::CompressedList),
            3 => Ok(Self::CompressedEdges),
            _ => unreachable!(),
        }
    }

    pub fn wire_value(self) -> u8 {
        match self {
            Self::None => 0,
            Self::UncompressedEdges => 1,
            Self::CompressedList => 2,
            Self::CompressedEdges => 3,
        }
    }

    /// Choose the most bandwidth-efficient format for a given ID set.
    pub fn best_for(ids: &[u32]) -> Self {
        if ids.is_empty() {
            return Self::None;
        }
        if is_contiguous(ids) {
            return Self::None;
        }
        // For now always use CompressedList for non-contiguous sets.
        // A production implementation would compare wire sizes and pick the minimum.
        Self::CompressedList
    }
}

// ── Encoding ─────────────────────────────────────────────────────────────

/// Encode source IDs into a raw byte vector according to `format`.
/// The returned bytes are NOT padded; the caller pads to 32-bit boundary.
pub fn encode(ids: &[u32], format: IdStorageFormat) -> Vec<u8> {
    match format {
        IdStorageFormat::None => {
            // No bit field needed — IDs are implicit from first_src_id + nb_ids.
            Vec::new()
        }
        IdStorageFormat::CompressedList => encode_compressed_list(ids),
        IdStorageFormat::UncompressedEdges => encode_uncompressed_edges(ids),
        IdStorageFormat::CompressedEdges => encode_compressed_edges(ids),
    }
}

fn encode_compressed_list(ids: &[u32]) -> Vec<u8> {
    // Delta-encode: write differences between consecutive IDs as variable-length
    // bit fields.  For simplicity we use 16-bit deltas (good enough for typical
    // windows ≤ 65535 apart).
    let mut writer = BitWriter::new();
    let mut prev = ids[0];
    // First ID is encoded in FIRST_SOURCE_ID field, so we only write deltas for the rest.
    for &id in ids.iter().skip(1) {
        let delta = id.wrapping_sub(prev);
        // Encode as 16-bit little-endian varint for simplicity
        writer.write_bits(delta, 16);
        prev = id;
    }
    writer.finish()
}

fn encode_uncompressed_edges(ids: &[u32]) -> Vec<u8> {
    // Encode run boundaries: for each contiguous block write (start, end-exclusive)
    // as 32-bit values relative to first_src_id.
    let blocks = to_blocks(ids);
    let mut out = Vec::with_capacity(blocks.len() * 8);
    let base = ids[0];
    for (start, end) in blocks {
        out.extend_from_slice(&(start - base).to_be_bytes());
        out.extend_from_slice(&(end - base).to_be_bytes());
    }
    out
}

fn encode_compressed_edges(ids: &[u32]) -> Vec<u8> {
    let blocks = to_blocks(ids);
    let mut writer = BitWriter::new();
    let base = ids[0];
    let mut prev = 0u32;
    for (start, end) in blocks {
        let d_start = (start - base) - prev;
        let d_len = end - start;
        writer.write_bits(d_start, 16);
        writer.write_bits(d_len, 16);
        prev = end - base;
    }
    writer.finish()
}

// ── Decoding ─────────────────────────────────────────────────────────────

/// Decode source IDs from the bit-field portion of an encoding vector.
///
/// Returns the decoded ID list and the number of bytes consumed from `buf`.
pub fn decode(
    buf: &[u8],
    nb_ids: usize,
    first_src_id: u32,
    format: IdStorageFormat,
) -> Result<(Vec<u32>, usize)> {
    match format {
        IdStorageFormat::None => {
            let ids = (0..nb_ids as u32).map(|i| first_src_id + i).collect();
            Ok((ids, 0))
        }
        IdStorageFormat::CompressedList => decode_compressed_list(buf, nb_ids, first_src_id),
        IdStorageFormat::UncompressedEdges => decode_uncompressed_edges(buf, nb_ids, first_src_id),
        IdStorageFormat::CompressedEdges => decode_compressed_edges(buf, nb_ids, first_src_id),
    }
}

fn decode_compressed_list(
    buf: &[u8],
    nb_ids: usize,
    first_src_id: u32,
) -> Result<(Vec<u32>, usize)> {
    let mut ids = Vec::with_capacity(nb_ids);
    ids.push(first_src_id);
    if nb_ids == 1 {
        return Ok((ids, 0));
    }

    let needed = (nb_ids - 1) * 2; // 16 bits per delta
    if buf.len() < needed {
        return Err(DelpError::MalformedEncodingVector {
            reason: "CompressedList buffer too short",
        });
    }

    let mut reader = BitReader::new(buf);
    let mut prev = first_src_id;
    for _ in 1..nb_ids {
        let delta = reader.read_bits(16);
        prev += delta;
        ids.push(prev);
    }
    let bytes_read = reader.bytes_consumed();
    Ok((ids, bytes_read))
}

fn decode_uncompressed_edges(
    buf: &[u8],
    nb_ids: usize,
    first_src_id: u32,
) -> Result<(Vec<u32>, usize)> {
    // Each block = 2 × u32 (start_delta, end_delta)
    let mut ids = Vec::with_capacity(nb_ids);
    let mut off = 0usize;
    while ids.len() < nb_ids {
        if buf.len() < off + 8 {
            return Err(DelpError::MalformedEncodingVector {
                reason: "UncompressedEdges truncated",
            });
        }
        let start_delta = u32::from_be_bytes(buf[off..off + 4].try_into().unwrap());
        let end_delta = u32::from_be_bytes(buf[off + 4..off + 8].try_into().unwrap());
        off += 8;
        let start = first_src_id + start_delta;
        let end = first_src_id + end_delta;
        for id in start..end {
            ids.push(id);
        }
    }
    ids.truncate(nb_ids);
    Ok((ids, off))
}

fn decode_compressed_edges(
    buf: &[u8],
    nb_ids: usize,
    first_src_id: u32,
) -> Result<(Vec<u32>, usize)> {
    let mut ids = Vec::with_capacity(nb_ids);
    let mut reader = BitReader::new(buf);
    let mut abs = 0u32;
    while ids.len() < nb_ids {
        if reader.remaining_bits() < 32 {
            return Err(DelpError::MalformedEncodingVector {
                reason: "CompressedEdges truncated",
            });
        }
        let d_start = reader.read_bits(16);
        let d_len = reader.read_bits(16);
        let start = first_src_id + abs + d_start;
        abs = start - first_src_id + d_len;
        for id in start..start + d_len {
            ids.push(id);
        }
    }
    ids.truncate(nb_ids);
    Ok((ids, reader.bytes_consumed()))
}

// ── Helpers ───────────────────────────────────────────────────────────────

fn is_contiguous(ids: &[u32]) -> bool {
    ids.windows(2).all(|w| w[1] == w[0] + 1)
}

/// Split a sorted ID slice into contiguous blocks of (start, end_exclusive).
fn to_blocks(ids: &[u32]) -> Vec<(u32, u32)> {
    if ids.is_empty() {
        return Vec::new();
    }
    let mut blocks = Vec::new();
    let mut start = ids[0];
    let mut prev = ids[0];
    for &id in ids.iter().skip(1) {
        if id != prev + 1 {
            blocks.push((start, prev + 1));
            start = id;
        }
        prev = id;
    }
    blocks.push((start, prev + 1));
    blocks
}

// ── BitWriter / BitReader ─────────────────────────────────────────────────

struct BitWriter {
    buf: Vec<u8>,
    cur_byte: u8,
    bit_pos: u8, // bits written into cur_byte (0..8)
}

impl BitWriter {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            cur_byte: 0,
            bit_pos: 0,
        }
    }

    fn write_bits(&mut self, value: u32, nbits: u8) {
        for i in (0..nbits).rev() {
            let bit = ((value >> i) & 1) as u8;
            self.cur_byte = (self.cur_byte << 1) | bit;
            self.bit_pos += 1;
            if self.bit_pos == 8 {
                self.buf.push(self.cur_byte);
                self.cur_byte = 0;
                self.bit_pos = 0;
            }
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.bit_pos > 0 {
            self.buf.push(self.cur_byte << (8 - self.bit_pos));
        }
        self.buf
    }
}

struct BitReader<'a> {
    buf: &'a [u8],
    byte_i: usize,
    bit_pos: u8, // bits consumed in current byte (0..8)
}

impl<'a> BitReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self {
            buf,
            byte_i: 0,
            bit_pos: 0,
        }
    }

    fn read_bits(&mut self, nbits: u8) -> u32 {
        let mut val = 0u32;
        for _ in 0..nbits {
            if self.byte_i >= self.buf.len() {
                break;
            }
            let bit = (self.buf[self.byte_i] >> (7 - self.bit_pos)) & 1;
            val = (val << 1) | bit as u32;
            self.bit_pos += 1;
            if self.bit_pos == 8 {
                self.byte_i += 1;
                self.bit_pos = 0;
            }
        }
        val
    }

    fn bytes_consumed(&self) -> usize {
        if self.bit_pos > 0 {
            self.byte_i + 1
        } else {
            self.byte_i
        }
    }

    fn remaining_bits(&self) -> usize {
        let remaining_bytes = self.buf.len().saturating_sub(self.byte_i);
        remaining_bytes * 8 - self.bit_pos as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(ids: &[u32], format: IdStorageFormat) {
        let encoded = encode(ids, format);
        let (decoded, _) = decode(&encoded, ids.len(), ids[0], format).unwrap();
        assert_eq!(decoded, ids, "round-trip failed for {format:?}");
    }

    #[test]
    fn none_format() {
        let ids: Vec<u32> = (5..15).collect();
        round_trip(&ids, IdStorageFormat::None);
    }

    #[test]
    fn compressed_list_non_contiguous() {
        let ids = vec![1u32, 3, 7, 8, 15, 100];
        round_trip(&ids, IdStorageFormat::CompressedList);
    }

    #[test]
    fn uncompressed_edges() {
        let ids = vec![1u32, 2, 3, 10, 11, 20];
        round_trip(&ids, IdStorageFormat::UncompressedEdges);
    }

    #[test]
    fn compressed_edges() {
        let ids = vec![0u32, 1, 2, 5, 6, 7, 20];
        round_trip(&ids, IdStorageFormat::CompressedEdges);
    }

    #[test]
    fn best_for_contiguous() {
        let ids: Vec<u32> = (10..20).collect();
        assert_eq!(IdStorageFormat::best_for(&ids), IdStorageFormat::None);
    }

    #[test]
    fn best_for_non_contiguous() {
        let ids = vec![1u32, 3, 5];
        assert_ne!(IdStorageFormat::best_for(&ids), IdStorageFormat::None);
    }
}
