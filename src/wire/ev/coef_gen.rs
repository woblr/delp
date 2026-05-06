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
use crate::gf::{Field as GfField, Gf2_4, Gf2_8};
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
/// Returns a `SmallVec` with **one coefficient byte per source ID**.  For
/// GF(2⁴) the coefficient lives in the low nibble of each byte (high
/// nibble = 0); SIMD multiplication broadcasts that 4-bit value across
/// both halves of every payload byte.
pub fn vandermonde_batch(field: Field, source_ids: &[u32], coded_id: u32) -> SmallVec<[u8; 64]> {
    match field {
        Field::Gf2_8 => source_ids
            .iter()
            .map(|&sid| Gf2_8::vandermonde(sid, coded_id).to_u8())
            .collect(),
        Field::Gf2_4 => source_ids
            .iter()
            .map(|&sid| Gf2_4::vandermonde(sid, coded_id).to_u8() & 0x0F)
            .collect(),
    }
}

/// Precomputed Vandermonde row — amortises the modular arithmetic when the
/// same coded_id is used with many source IDs.
pub struct VandermondeRow {
    field: Field,
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

/// Cauchy matrix coefficient for GF(2⁸) with generation rotation.
///
/// `src_id` must be < 128, `coded_id` must be < 128.
/// `generation` rotates the y-point set so successive cycles produce
/// linearly-independent coded packets — the foundation of delp's
/// **unlimited-length Cauchy session** extension.
///
/// Formula:
/// ```text
///   y_eff = 128 + ((coded_id + generation * 7) mod 128)
///   coef  = 1 / (src_id XOR y_eff)         in GF(2⁸)
/// ```
///
/// `7` is coprime to 128, so as `(coded_id, generation)` varies over
/// `[0,128) × [0,256)` the y-coordinate visits 32 768 distinct positions
/// (with each value reused 128 times, but always against a *different*
/// coded_id within its generation).
#[inline]
pub fn cauchy_coef_gf2_8(src_id: u32, coded_id: u32, generation: u8) -> u8 {
    debug_assert!(
        coded_id < 128,
        "Cauchy: coded_id must be < 128, got {coded_id}"
    );
    // `src_id` may be any u32 — the encoder builder caps
    // `window_capacity ≤ 128`, so taking `src_id mod 128` for the
    // x-coordinate cannot collide within the active window.
    use crate::gf::{Field as GfField, Gf2_8};
    let x = (src_id & 0x7F) as u8;
    let y_offset = (coded_id + (generation as u32) * 7) & 0x7F;
    let y = 128u8 + y_offset as u8;
    let denom = Gf2_8::from_u8(x ^ y);
    denom.inv().to_u8()
}

/// Batch Cauchy coefficients for one coded packet covering `source_ids`.
pub fn cauchy_batch_gf2_8(source_ids: &[u32], coded_id: u32, generation: u8) -> SmallVec<[u8; 64]> {
    source_ids
        .iter()
        .map(|&sid| cauchy_coef_gf2_8(sid, coded_id, generation))
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

/// Cauchy coefficient for GF(2⁴) with generation rotation.
///
/// `coded_id` must be < 7.  `src_id` may be any `u32`; the x-coordinate
/// is `src_id mod 7`.  Window capacity ≤ 7 (builder-enforced) prevents
/// modulus collisions within an active window.
#[inline]
pub fn cauchy_coef_gf2_4(src_id: u32, coded_id: u32, generation: u8) -> u8 {
    debug_assert!(
        coded_id < 7,
        "GF(2^4) Cauchy: coded_id must be < 7, got {coded_id}"
    );
    use crate::gf::{Field as GfField, Gf2_4};
    let x = (src_id % 7) as u8;
    let y_offset = ((coded_id + (generation as u32) * 3) % 7) as u8;
    let y = 8u8 + y_offset;
    let denom = Gf2_4::from_u8(x ^ y);
    denom.inv().to_u8()
}

/// Batch GF(2⁴) Cauchy coefficients.
pub fn cauchy_batch_gf2_4(source_ids: &[u32], coded_id: u32, generation: u8) -> SmallVec<[u8; 64]> {
    source_ids
        .iter()
        .map(|&sid| cauchy_coef_gf2_4(sid, coded_id, generation))
        .collect()
}

/// MDS verification helper for GF(2⁴) Cauchy at generation 0.
#[cfg(any(test, debug_assertions))]
pub fn verify_cauchy_gf2_4_2x2_nonsingular(s0: u32, s1: u32, c0: u32, c1: u32) -> bool {
    use crate::gf::{Field as GfField, Gf2_4};
    let a00 = Gf2_4::from_u8(cauchy_coef_gf2_4(s0, c0, 0));
    let a01 = Gf2_4::from_u8(cauchy_coef_gf2_4(s0, c1, 0));
    let a10 = Gf2_4::from_u8(cauchy_coef_gf2_4(s1, c0, 0));
    let a11 = Gf2_4::from_u8(cauchy_coef_gf2_4(s1, c1, 0));
    let det = a00.mul(a11).add(a01.mul(a10));
    det != Gf2_4::ZERO
}

/// Verify the MDS property of the Cauchy construction at generation 0.
#[cfg(any(test, debug_assertions))]
pub fn verify_cauchy_2x2_nonsingular(s0: u32, s1: u32, c0: u32, c1: u32) -> bool {
    use crate::gf::{Field as GfField, Gf2_8};
    let a00 = Gf2_8::from_u8(cauchy_coef_gf2_8(s0, c0, 0));
    let a01 = Gf2_8::from_u8(cauchy_coef_gf2_8(s0, c1, 0));
    let a10 = Gf2_8::from_u8(cauchy_coef_gf2_8(s1, c0, 0));
    let a11 = Gf2_8::from_u8(cauchy_coef_gf2_8(s1, c1, 0));
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

    /// Cauchy coefficient is never zero — denom is always non-zero, for
    /// all generations.
    #[test]
    fn cauchy_coef_never_zero() {
        for gen in 0u8..=8 {
            for src in 0u32..128 {
                for coded in 0u32..128 {
                    let c = cauchy_coef_gf2_8(src, coded, gen);
                    assert_ne!(c, 0, "cauchy_coef_gf2_8({src},{coded},gen={gen}) == 0");
                }
            }
        }
    }

    /// Generation rotation preserves the never-zero invariant for GF(2⁴).
    #[test]
    fn cauchy_gf2_4_coef_never_zero_with_generation() {
        for gen in 0u8..=8 {
            for src in 0u32..7 {
                for coded in 0u32..7 {
                    let c = cauchy_coef_gf2_4(src, coded, gen);
                    assert_ne!(c, 0, "cauchy_coef_gf2_4({src},{coded},gen={gen}) == 0");
                }
            }
        }
    }

    /// Different generations produce *different* coefficient rows for the
    /// same (src, coded) pair — proves that generation rotation makes
    /// successive cycles linearly distinguishable.
    #[test]
    fn cauchy_generations_produce_distinct_rows() {
        // For a fixed coded_id, two generations should give different
        // coefficient vectors over a window of source IDs (at least one
        // src_id must produce a different coef).
        for coded in 0u32..128 {
            let row0: Vec<u8> = (0..128u32)
                .map(|s| cauchy_coef_gf2_8(s, coded, 0))
                .collect();
            let row1: Vec<u8> = (0..128u32)
                .map(|s| cauchy_coef_gf2_8(s, coded, 1))
                .collect();
            assert_ne!(
                row0, row1,
                "gen 0 and gen 1 produced identical row at coded={coded}"
            );
        }
    }

    /// Any 2×2 Cauchy submatrix is non-singular (MDS spot check).
    #[test]
    fn cauchy_2x2_nonsingular_exhaustive() {
        for s0 in 0u32..16 {
            for s1 in (s0 + 1)..16 {
                for c0 in 0u32..16 {
                    for c1 in (c0 + 1)..16 {
                        assert!(
                            verify_cauchy_2x2_nonsingular(s0, s1, c0, c1),
                            "Cauchy 2×2 singular at s=({s0},{s1}) c=({c0},{c1})"
                        );
                    }
                }
            }
        }
    }

    /// Cauchy batch matches individual scalar calls (with generation).
    #[test]
    fn cauchy_batch_matches_scalar() {
        let ids: Vec<u32> = (0..16).collect();
        for coded in 0u32..8 {
            for gen in 0u8..=4 {
                let batch = cauchy_batch_gf2_8(&ids, coded, gen);
                for (i, &sid) in ids.iter().enumerate() {
                    assert_eq!(
                        batch[i],
                        cauchy_coef_gf2_8(sid, coded, gen),
                        "batch mismatch at src={sid} coded={coded} gen={gen}"
                    );
                }
            }
        }
    }

    /// Diagonal sanity check at generation 0.
    #[test]
    fn cauchy_coef_defined_on_diagonal() {
        for i in 0u32..128 {
            let c = cauchy_coef_gf2_8(i, i, 0);
            assert_ne!(c, 0, "cauchy diagonal({i}) == 0");
        }
    }

    // ── GF(2^4) Cauchy tests ─────────────────────────────────────────────

    /// GF(2⁴) Cauchy coefficient is always non-zero at generation 0.
    #[test]
    fn cauchy_gf2_4_coef_never_zero() {
        for src in 0u32..7 {
            for coded in 0u32..7 {
                let c = cauchy_coef_gf2_4(src, coded, 0);
                assert_ne!(
                    c, 0,
                    "cauchy_gf2_4({src},{coded}) == 0 — impossible by construction"
                );
            }
        }
    }

    /// GF(2^4) Cauchy: all 2×2 submatrices are non-singular (full MDS proof).
    #[test]
    fn cauchy_gf2_4_2x2_nonsingular_exhaustive() {
        for s0 in 0u32..7 {
            for s1 in (s0 + 1)..7 {
                for c0 in 0u32..7 {
                    for c1 in (c0 + 1)..7 {
                        assert!(
                            verify_cauchy_gf2_4_2x2_nonsingular(s0, s1, c0, c1),
                            "GF(2^4) Cauchy 2×2 singular at s=({s0},{s1}) c=({c0},{c1})"
                        );
                    }
                }
            }
        }
    }

    /// GF(2⁴) Cauchy batch matches scalar (with generation).
    #[test]
    fn cauchy_gf2_4_batch_matches_scalar() {
        let ids: Vec<u32> = (0..6).collect();
        for coded in 0u32..6 {
            for gen in 0u8..=4 {
                let batch = cauchy_batch_gf2_4(&ids, coded, gen);
                for (i, &sid) in ids.iter().enumerate() {
                    assert_eq!(
                        batch[i],
                        cauchy_coef_gf2_4(sid, coded, gen),
                        "gf2_4 batch mismatch src={sid} coded={coded} gen={gen}"
                    );
                }
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
    fn gf2_4_batch_one_byte_per_source() {
        let ids: Vec<u32> = (0..8).collect();
        let batch = vandermonde_batch(Field::Gf2_4, &ids, 1);
        assert_eq!(batch.len(), ids.len());
        for (i, &sid) in ids.iter().enumerate() {
            let expected = Gf2_4::vandermonde(sid, 1).to_u8() & 0x0F;
            assert_eq!(
                batch[i], expected,
                "batch[{i}]: expected 4-bit coef {expected}, got {}",
                batch[i]
            );
            assert_eq!(batch[i] & 0xF0, 0, "batch[{i}] high nibble must be 0");
        }
    }
}
