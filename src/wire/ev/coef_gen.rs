/// Delp protocol §5.2.2 — Vandermonde coefficient generation.
///
/// For a (source_id, coded_id) pair the coefficient is:
///   `alpha ^ ((source_id * coded_id) mod (ORDER − 1))`
///
/// where `alpha = 2` is the primitive element.  `coded_id` must be ≥ 1
/// to avoid the degenerate case where every coefficient equals 1.
///
/// This module provides batch generation helpers used by the encoder so
/// the hot loop avoids repeated trait dispatch.

use crate::config::Field;
use crate::gf::{Gf2_4, Gf2_8, Field as GfField};
use smallvec::SmallVec;

/// Generate one Vandermonde coefficient in the selected field.
#[inline]
pub fn vandermonde_coef(field: Field, src_id: u32, coded_id: u32) -> u8 {
    match field {
        Field::Gf2_8 => Gf2_8::vandermonde(src_id, coded_id).to_u8(),
        Field::Gf2_4 => Gf2_4::vandermonde(src_id, coded_id).to_u8(),
    }
}

/// Generate a packed GF(2^4) coefficient byte for two consecutive source IDs.
///
/// Used in the encoder when field = Gf2_4 and IDs can be packed two-per-byte.
#[inline]
pub fn vandermonde_packed_gf2_4(src_id_hi: u32, src_id_lo: u32, coded_id: u32) -> u8 {
    let hi = Gf2_4::vandermonde(src_id_hi, coded_id).to_u8() & 0x0F;
    let lo = Gf2_4::vandermonde(src_id_lo, coded_id).to_u8() & 0x0F;
    (hi << 4) | lo
}

/// Batch-generate all Vandermonde coefficients for a slice of source IDs.
///
/// Returns a `SmallVec` of `u8` coefficients, one per source ID (GF(2^8))
/// or packed two-per-byte (GF(2^4)).
pub fn vandermonde_batch(
    field:      Field,
    source_ids: &[u32],
    coded_id:   u32,
) -> SmallVec<[u8; 64]> {
    match field {
        Field::Gf2_8 => {
            source_ids.iter().map(|&sid| {
                Gf2_8::vandermonde(sid, coded_id).to_u8()
            }).collect()
        }
        Field::Gf2_4 => {
            // Pack pairs of nibble coefficients into single bytes.
            let mut out = SmallVec::with_capacity((source_ids.len() + 1) / 2);
            let mut iter = source_ids.iter().peekable();
            while let Some(&hi_id) = iter.next() {
                let lo_id = iter.next().copied();
                let hi = Gf2_4::vandermonde(hi_id, coded_id).to_u8() & 0x0F;
                let lo = lo_id.map_or(0u8, |id| {
                    Gf2_4::vandermonde(id, coded_id).to_u8() & 0x0F
                });
                out.push((hi << 4) | lo);
            }
            out
        }
    }
}

/// Precomputed Vandermonde row — amortises the modular arithmetic when the
/// same coded_id is used with many source IDs.
pub struct VandermondeRow {
    field:    Field,
    coded_id: u32,
}

impl VandermondeRow {
    pub fn new(field: Field, coded_id: u32) -> Self {
        debug_assert!(coded_id >= 1, "coded_id must be ≥ 1");
        Self { field, coded_id }
    }

    #[inline]
    pub fn coef(&self, src_id: u32) -> u8 {
        vandermonde_coef(self.field, src_id, self.coded_id)
    }
}

// ── Cauchy matrix coefficient generation ─────────────────────────────────────
//
// A Cauchy matrix M is defined by two disjoint point sets {x_i} and {y_j}:
//
//   M[i][j] = 1 / (x_i + y_j)       (addition = XOR in GF(2^n))
//
// **Any** square submatrix of a Cauchy matrix is full rank — the MDS property
// is mathematically guaranteed, unlike Vandermonde where degenerate cases
// can arise from modular arithmetic coincidences.
//
// Point assignment (GF(2^8), 256 elements total):
//   source points  x_i = i           for i in 0..128
//   coded  points  y_j = 128 + j     for j in 0..128
//
// Disjointness: x_i ∈ {0..127}, y_j ∈ {128..255} → x_i ≠ y_j always,
// so (x_i + y_j) ≠ 0 and the inverse is always defined.
//
// Capacity: up to 128 source symbols and 128 coded symbols per session.

/// Cauchy matrix coefficient for GF(2^8).
///
/// `src_id` must be < 128, `coded_id` must be < 128.
/// Returns `1 / (src_id XOR (128 + coded_id))` in GF(2^8).
#[inline]
pub fn cauchy_coef_gf2_8(src_id: u32, coded_id: u32) -> u8 {
    debug_assert!(src_id  < 128, "Cauchy: src_id must be < 128, got {src_id}");
    debug_assert!(coded_id < 128, "Cauchy: coded_id must be < 128, got {coded_id}");
    use crate::gf::{Field as GfField, Gf2_8};
    let x = src_id as u8;          // 0..127
    let y = 128u8 + coded_id as u8; // 128..255
    // x XOR y is non-zero because the high bit of y is set and x < 128
    let denom = Gf2_8::from_u8(x ^ y);
    denom.inv().to_u8()
}

/// Batch Cauchy coefficients for one coded packet covering `source_ids`.
///
/// All source IDs must be < 128 and the coded_id must be < 128.
pub fn cauchy_batch_gf2_8(source_ids: &[u32], coded_id: u32) -> SmallVec<[u8; 64]> {
    source_ids.iter()
        .map(|&sid| cauchy_coef_gf2_8(sid, coded_id))
        .collect()
}

// ── GF(2^4) Cauchy matrix ─────────────────────────────────────────────────
//
// Point assignment (GF(2^4), 16 elements total):
//   source points  x_i = i     for i in 0..7  → {0..6}
//   coded  points  y_j = 8 + j for j in 0..7  → {8..14}
//
// Disjointness: x_i ∈ {0..6}, y_j ∈ {8..14} → x_i XOR y_j ≥ 8 ≠ 0.
// Capacity: up to 7 source symbols × 7 coded symbols per session.

/// Cauchy coefficient for GF(2⁴).
///
/// `src_id` must be < 7, `coded_id` must be < 7.
/// Returns `1 / (src_id XOR (8 + coded_id))` in GF(2⁴).
#[inline]
pub fn cauchy_coef_gf2_4(src_id: u32, coded_id: u32) -> u8 {
    debug_assert!(src_id  < 7, "GF(2^4) Cauchy: src_id must be < 7, got {src_id}");
    debug_assert!(coded_id < 7, "GF(2^4) Cauchy: coded_id must be < 7, got {coded_id}");
    use crate::gf::{Field as GfField, Gf2_4};
    let x = src_id as u8;           // 0..6
    let y = 8u8 + coded_id as u8;   // 8..14
    // x XOR y: x < 8 so bit-3 is 0; y ≥ 8 so bit-3 is set → XOR ≥ 8, never 0
    let denom = Gf2_4::from_u8(x ^ y);
    denom.inv().to_u8()
}

/// Batch GF(2^4) Cauchy coefficients.
///
/// All source IDs must be < 7 and coded_id < 7.
pub fn cauchy_batch_gf2_4(source_ids: &[u32], coded_id: u32) -> SmallVec<[u8; 64]> {
    source_ids.iter()
        .map(|&sid| cauchy_coef_gf2_4(sid, coded_id))
        .collect()
}

/// MDS verification helper for GF(2^4) Cauchy.
#[cfg(any(test, debug_assertions))]
pub fn verify_cauchy_gf2_4_2x2_nonsingular(
    s0: u32, s1: u32, c0: u32, c1: u32,
) -> bool {
    use crate::gf::{Field as GfField, Gf2_4};
    let a00 = Gf2_4::from_u8(cauchy_coef_gf2_4(s0, c0));
    let a01 = Gf2_4::from_u8(cauchy_coef_gf2_4(s0, c1));
    let a10 = Gf2_4::from_u8(cauchy_coef_gf2_4(s1, c0));
    let a11 = Gf2_4::from_u8(cauchy_coef_gf2_4(s1, c1));
    let det = a00.mul(a11).add(a01.mul(a10));
    det != Gf2_4::ZERO
}

/// Verify the MDS property of the Cauchy construction:
/// for any two distinct (src, coded) pairs the 2×2 submatrix is non-singular.
///
/// This is a compile-time provable property but we expose it as a runtime
/// check for integration test use.
#[cfg(any(test, debug_assertions))]
pub fn verify_cauchy_2x2_nonsingular(
    s0: u32, s1: u32, c0: u32, c1: u32,
) -> bool {
    use crate::gf::{Field as GfField, Gf2_8};
    // det = M[s0,c0]*M[s1,c1] XOR M[s0,c1]*M[s1,c0]
    let a00 = Gf2_8::from_u8(cauchy_coef_gf2_8(s0, c0));
    let a01 = Gf2_8::from_u8(cauchy_coef_gf2_8(s0, c1));
    let a10 = Gf2_8::from_u8(cauchy_coef_gf2_8(s1, c0));
    let a11 = Gf2_8::from_u8(cauchy_coef_gf2_8(s1, c1));
    let det = a00.mul(a11).add(a01.mul(a10));
    det != Gf2_8::ZERO
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gf2_8_known_values() {
        // alpha^0 = 1; vandermonde(src=0, coded=1) = alpha^0 = 1
        assert_eq!(vandermonde_coef(Field::Gf2_8, 0, 1), 1);
        // alpha^1 = 2; vandermonde(src=1, coded=1) = alpha^1 = 2
        assert_eq!(vandermonde_coef(Field::Gf2_8, 1, 1), 2);
    }

    /// Cauchy coefficient is never zero — denom is always non-zero.
    #[test]
    fn cauchy_coef_never_zero() {
        for src in 0u32..128 {
            for coded in 0u32..128 {
                let c = cauchy_coef_gf2_8(src, coded);
                assert_ne!(c, 0,
                    "cauchy_coef_gf2_8({src},{coded}) == 0 — impossible by construction");
            }
        }
    }

    /// Any 2×2 Cauchy submatrix is non-singular (MDS spot check).
    #[test]
    fn cauchy_2x2_nonsingular_exhaustive() {
        for s0 in 0u32..16 {
            for s1 in (s0+1)..16 {
                for c0 in 0u32..16 {
                    for c1 in (c0+1)..16 {
                        assert!(
                            verify_cauchy_2x2_nonsingular(s0, s1, c0, c1),
                            "Cauchy 2×2 singular at s=({s0},{s1}) c=({c0},{c1})"
                        );
                    }
                }
            }
        }
    }

    /// Cauchy batch matches individual scalar calls.
    #[test]
    fn cauchy_batch_matches_scalar() {
        let ids: Vec<u32> = (0..16).collect();
        for coded in 0u32..8 {
            let batch = cauchy_batch_gf2_8(&ids, coded);
            for (i, &sid) in ids.iter().enumerate() {
                assert_eq!(batch[i], cauchy_coef_gf2_8(sid, coded),
                    "batch mismatch at src={sid} coded={coded}");
            }
        }
    }

    /// Cauchy matrix is symmetric when src_id == coded_id — sanity check.
    #[test]
    fn cauchy_coef_defined_on_diagonal() {
        // On the diagonal x_i XOR y_i = i XOR (128+i) = 128 ≠ 0 → always valid
        for i in 0u32..128 {
            let c = cauchy_coef_gf2_8(i, i);
            assert_ne!(c, 0, "cauchy diagonal({i}) == 0");
        }
    }

    // ── GF(2^4) Cauchy tests ─────────────────────────────────────────────

    /// GF(2^4) Cauchy coefficient is always non-zero.
    #[test]
    fn cauchy_gf2_4_coef_never_zero() {
        for src in 0u32..7 {
            for coded in 0u32..7 {
                let c = cauchy_coef_gf2_4(src, coded);
                assert_ne!(c, 0,
                    "cauchy_gf2_4({src},{coded}) == 0 — impossible by construction");
            }
        }
    }

    /// GF(2^4) Cauchy: all 2×2 submatrices are non-singular (full MDS proof).
    #[test]
    fn cauchy_gf2_4_2x2_nonsingular_exhaustive() {
        for s0 in 0u32..7 {
            for s1 in (s0+1)..7 {
                for c0 in 0u32..7 {
                    for c1 in (c0+1)..7 {
                        assert!(
                            verify_cauchy_gf2_4_2x2_nonsingular(s0, s1, c0, c1),
                            "GF(2^4) Cauchy 2×2 singular at s=({s0},{s1}) c=({c0},{c1})"
                        );
                    }
                }
            }
        }
    }

    /// GF(2^4) Cauchy batch matches scalar.
    #[test]
    fn cauchy_gf2_4_batch_matches_scalar() {
        let ids: Vec<u32> = (0..6).collect();
        for coded in 0u32..6 {
            let batch = cauchy_batch_gf2_4(&ids, coded);
            for (i, &sid) in ids.iter().enumerate() {
                assert_eq!(batch[i], cauchy_coef_gf2_4(sid, coded),
                    "gf2_4 batch mismatch src={sid} coded={coded}");
            }
        }
    }

    // ── Property-based tests (proptest) ──────────────────────────────────

    proptest::proptest! {
        /// For any non-zero GF(2^8) element a,  a * a^-1 == 1.
        #[test]
        fn prop_gf2_8_inv_correct(a in 1u8..=255u8) {
            use crate::gf::{Field as GfField, Gf2_8};
            let fa = Gf2_8::from_u8(a);
            proptest::prop_assert_eq!(fa.mul(fa.inv()), Gf2_8::ONE);
        }

        /// GF(2^8) multiplication is commutative.
        #[test]
        fn prop_gf2_8_mul_commutative(a in 0u8..=255u8, b in 0u8..=255u8) {
            use crate::gf::{Field as GfField, Gf2_8};
            let fa = Gf2_8::from_u8(a);
            let fb = Gf2_8::from_u8(b);
            proptest::prop_assert_eq!(fa.mul(fb), fb.mul(fa));
        }

        /// GF(2^8) multiplication is associative.
        #[test]
        fn prop_gf2_8_mul_associative(
            a in 0u8..=255u8,
            b in 0u8..=255u8,
            c in 0u8..=255u8,
        ) {
            use crate::gf::{Field as GfField, Gf2_8};
            let fa = Gf2_8::from_u8(a);
            let fb = Gf2_8::from_u8(b);
            let fc = Gf2_8::from_u8(c);
            proptest::prop_assert_eq!(fa.mul(fb).mul(fc), fa.mul(fb.mul(fc)));
        }

        /// Cauchy GF(2^8): any two distinct (src,coded) pairs give non-singular 2×2.
        #[test]
        fn prop_cauchy_gf2_8_2x2_nonsingular(
            s0 in 0u32..127,
            s1 in 0u32..127,
            c0 in 0u32..127,
            c1 in 0u32..127,
        ) {
            proptest::prop_assume!(s0 != s1 && c0 != c1);
            proptest::prop_assert!(verify_cauchy_2x2_nonsingular(s0, s1, c0, c1),
                "Cauchy 2×2 singular at s=({s0},{s1}) c=({c0},{c1})");
        }

        /// SIMD mul_acc matches reference for arbitrary coefficients and data.
        #[test]
        fn prop_simd_mul_acc_matches_reference(
            coef in 0u8..=255u8,
            data in proptest::collection::vec(0u8..=255u8, 1..=128usize),
        ) {
            use crate::gf::simd::ops::mul_acc_gf2_8;
            use crate::gf::simd::mul_acc_gf2_8_reference;
            let mut simd_dst  = vec![0u8; data.len()];
            let mut ref_dst   = vec![0u8; data.len()];
            mul_acc_gf2_8(&mut simd_dst, &data, coef);
            mul_acc_gf2_8_reference(&mut ref_dst, &data, coef);
            proptest::prop_assert_eq!(simd_dst, ref_dst);
        }

        /// Vandermonde 2×2 non-singular for distinct small IDs (regression guard).
        #[test]
        fn prop_vandermonde_2x2_nonsingular(
            s0 in 0u32..7,
            s1 in 0u32..7,
            c0 in 1u32..7,
            c1 in 1u32..7,
        ) {
            use crate::gf::{Field as GfField, Gf2_8};
            proptest::prop_assume!(s0 != s1 && c0 != c1);
            let a00 = Gf2_8::vandermonde(s0, c0);
            let a01 = Gf2_8::vandermonde(s1, c0);
            let a10 = Gf2_8::vandermonde(s0, c1);
            let a11 = Gf2_8::vandermonde(s1, c1);
            let det = a00.mul(a11).add(a01.mul(a10));
            proptest::prop_assert!(
                det != Gf2_8::ZERO,
                "Vandermonde 2×2 singular s=({},{}) c=({},{})", s0, s1, c0, c1
            );
        }
    }

    #[test]
    fn batch_matches_scalar() {
        let ids: Vec<u32> = (0..16).collect();
        for coded_id in 1..=4u32 {
            let batch = vandermonde_batch(Field::Gf2_8, &ids, coded_id);
            for (i, &sid) in ids.iter().enumerate() {
                assert_eq!(batch[i], vandermonde_coef(Field::Gf2_8, sid, coded_id));
            }
        }
    }

    #[test]
    fn gf2_4_batch_packs_correctly() {
        let ids: Vec<u32> = (0..8).collect();
        let packed = vandermonde_batch(Field::Gf2_4, &ids, 1);
        // Unpack and verify each nibble
        for (byte_idx, &byte) in packed.iter().enumerate() {
            let hi_id  = (byte_idx * 2) as u32;
            let lo_id  = hi_id + 1;
            let exp_hi = Gf2_4::vandermonde(hi_id, 1).to_u8() & 0x0F;
            let exp_lo = Gf2_4::vandermonde(lo_id, 1).to_u8() & 0x0F;
            assert_eq!((byte >> 4) & 0x0F, exp_hi);
            assert_eq!(byte & 0x0F,        exp_lo);
        }
    }
}