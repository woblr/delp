pub mod gf2_4;
pub mod gf2_8;
pub(crate) mod simd;
pub(crate) mod tables;

pub use gf2_4::Gf2_4;
pub use gf2_8::Gf2_8;

/// Sealed trait for Galois field elements.
///
/// Only `Gf2_4` and `Gf2_8` implement this trait.  External crates cannot
/// add new implementations, which allows the compiler to generate
/// exhaustive monomorphisations without a vtable.
pub trait Field: sealed::Sealed + Copy + Eq + core::fmt::Debug {
    /// Number of elements in the field (16 or 256).
    const ORDER: usize;
    /// Bits needed to represent one coefficient (4 or 8).
    const COEF_BITS: u8;
    /// Additive identity.
    const ZERO: Self;
    /// Multiplicative identity.
    const ONE: Self;

    // ── Arithmetic ──────────────────────────────────────────────────────
    /// Addition (= XOR in characteristic-2 fields).
    fn add(self, rhs: Self) -> Self;
    /// Subtraction — identical to addition in characteristic-2.
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self { self.add(rhs) }
    /// Multiplication via precomputed log/exp tables.
    fn mul(self, rhs: Self) -> Self;
    /// Multiplicative inverse via log/exp tables.
    ///
    /// # Panics
    /// Panics in debug builds if `self` is the zero element.
    fn inv(self) -> Self;
    /// Division: `self * rhs⁻¹`.
    #[inline(always)]
    fn div(self, rhs: Self) -> Self { self.mul(rhs.inv()) }

    // ── Generators ──────────────────────────────────────────────────────
    /// `alpha ^ exp` where `alpha` is the primitive element (= 2).
    fn alpha_pow(exp: u32) -> Self;
    /// Vandermonde coefficient for a (src_id, coded_id) pair.
    ///
    /// Defined as `alpha ^ ((src_id * coded_id) mod (ORDER − 1))`.
    /// `coded_id` must be ≥ 1 to avoid the degenerate all-ones case.
    #[inline]
    fn vandermonde(src_id: u32, coded_id: u32) -> Self {
        debug_assert!(coded_id >= 1, "coded_id must be ≥ 1");
        let modulus = (Self::ORDER - 1) as u64;
        let exp = (src_id as u64 * coded_id as u64) % modulus;
        Self::alpha_pow(exp as u32)
    }

    // ── Byte conversion ──────────────────────────────────────────────────
    fn from_u8(v: u8) -> Self;
    fn to_u8(self) -> u8;
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::Gf2_4 {}
    impl Sealed for super::Gf2_8 {}
}

#[cfg(test)]
mod tests {
    use super::*;

    fn axioms<F: Field>() {
        let one  = F::ONE;
        let zero = F::ZERO;

        // additive identity
        for raw in 0..(F::ORDER as u8) {
            let a = F::from_u8(raw);
            assert_eq!(a.add(zero), a, "a + 0 == a failed for {raw}");
            assert_eq!(a.sub(a),   zero, "a - a == 0 failed for {raw}");
        }

        // multiplicative identity
        for raw in 0..(F::ORDER as u8) {
            let a = F::from_u8(raw);
            assert_eq!(a.mul(one), a, "a * 1 == a failed for {raw}");
        }

        // multiplicative inverse (non-zero)
        for raw in 1..(F::ORDER as u8) {
            let a = F::from_u8(raw);
            assert_eq!(a.mul(a.inv()), one, "a * a⁻¹ == 1 failed for {raw}");
        }

        // distributivity: a*(b+c) == a*b + a*c
        for a in 0..(F::ORDER as u8) {
            for b in 0..(F::ORDER as u8) {
                for c in 0..(F::ORDER as u8) {
                    let fa = F::from_u8(a);
                    let fb = F::from_u8(b);
                    let fc = F::from_u8(c);
                    assert_eq!(
                        fa.mul(fb.add(fc)),
                        fa.mul(fb).add(fa.mul(fc)),
                        "distributivity failed for ({a},{b},{c})"
                    );
                }
            }
        }
    }

    #[test] fn gf2_4_axioms() { axioms::<Gf2_4>(); }
    #[test] fn gf2_8_axioms() { axioms::<Gf2_8>(); }

    fn vandermonde_check<F: Field>() {
        // coded_id=1 → coefficients are alpha^(src_id * 1) = alpha^src_id
        for src in 0..8u32 {
            let coef = F::vandermonde(src, 1);
            assert_eq!(coef, F::alpha_pow(src % (F::ORDER as u32 - 1)));
        }
    }

    #[test] fn vandermonde_gf2_4() { vandermonde_check::<Gf2_4>(); }
    #[test] fn vandermonde_gf2_8() { vandermonde_check::<Gf2_8>(); }

    // ── Exhaustive correctness proofs ────────────────────────────────────

    /// Every element a satisfies: a * a^(-1) = 1  AND  a^(-1) * a = 1.
    /// Verified for ALL non-zero elements in the field.
    #[test]
    fn gf2_8_inv_exhaustive() {
        for v in 1u8..=255 {
            let a   = Gf2_8::from_u8(v);
            let inv = a.inv();
            assert_eq!(a.mul(inv), Gf2_8::ONE,
                "GF(2^8): {v} * inv({v}) != 1  [inv={:?}]", inv);
            assert_eq!(inv.mul(a), Gf2_8::ONE,
                "GF(2^8): inv({v}) * {v} != 1");
        }
    }

    /// All 256×256 GF(2^8) multiplications satisfy commutativity,
    /// zero absorption, and consistency with the log/exp definition.
    #[test]
    fn gf2_8_mul_exhaustive() {
        use super::tables::{GF2_8_EXP, GF2_8_LOG};
        for a in 0u16..256 {
            for b in 0u16..256 {
                let fa = Gf2_8::from_u8(a as u8);
                let fb = Gf2_8::from_u8(b as u8);
                let got = fa.mul(fb).to_u8();

                // Reference: direct log/exp computation
                let expected: u8 = if a == 0 || b == 0 {
                    0
                } else {
                    GF2_8_EXP[GF2_8_LOG[a as usize] as usize
                             + GF2_8_LOG[b as usize] as usize]
                };
                assert_eq!(got, expected, "GF(2^8) mul({a},{b}) mismatch");

                // Commutativity
                assert_eq!(fa.mul(fb), fb.mul(fa),
                    "GF(2^8) mul not commutative at ({a},{b})");
            }
        }
    }

    /// LOG table sentinel: LOG[0] must be 0xFF, never a valid exponent.
    #[test]
    fn gf2_8_log_sentinel() {
        use super::tables::GF2_8_LOG;
        assert_eq!(GF2_8_LOG[0], 0xFF,
            "GF(2^8) LOG[0] must be 0xFF sentinel, got {}", GF2_8_LOG[0]);
    }

    /// EXP and LOG are true inverses: EXP[LOG[x]] == x for all x in 1..=255.
    #[test]
    fn gf2_8_exp_log_inverse() {
        use super::tables::{GF2_8_EXP, GF2_8_LOG};
        for x in 1u8..=255 {
            let log_x = GF2_8_LOG[x as usize] as usize;
            assert_eq!(GF2_8_EXP[log_x], x,
                "EXP[LOG[{x}]] != {x}  (got {})", GF2_8_EXP[log_x]);
        }
        for i in 0usize..255 {
            let exp_i = GF2_8_EXP[i] as usize;
            assert_eq!(GF2_8_LOG[exp_i], i as u8,
                "LOG[EXP[{i}]] != {i}  (got {})", GF2_8_LOG[exp_i]);
        }
    }

    /// GF(2^4): all non-zero elements have a valid inverse.
    #[test]
    fn gf2_4_inv_exhaustive() {
        for v in 1u8..16 {
            let a   = Gf2_4::from_u8(v);
            let inv = a.inv();
            assert_eq!(a.mul(inv), Gf2_4::ONE,
                "GF(2^4): {v} * inv({v}) != 1  [inv={}]", inv.to_u8());
        }
    }

    /// EXP[LOG[x]] == x for all non-zero GF(2^4) elements.
    #[test]
    fn gf2_4_exp_log_inverse() {
        use super::tables::{GF2_4_EXP, GF2_4_LOG};
        for x in 1u8..16 {
            let log_x = GF2_4_LOG[x as usize] as usize;
            assert!(log_x < 15, "LOG[{x}] out of range: {log_x}");
            assert_eq!(GF2_4_EXP[log_x], x,
                "EXP[LOG[{x}]] != {x}");
        }
    }

    /// Vandermonde matrix rank check: any k distinct coded packets over
    /// the same k source IDs must recover all k sources.
    ///
    /// We verify the 2-erasure and 3-erasure cases exhaustively for small
    /// windows in GF(2^8).
    #[test]
    fn vandermonde_mds_2_erasure_gf2_8() {
        // For any two distinct coded IDs c1 != c2 and two source IDs s0 != s1,
        // the 2×2 Vandermonde submatrix must be non-singular.
        //
        // M = | alpha^(s0*c1)  alpha^(s1*c1) |
        //     | alpha^(s0*c2)  alpha^(s1*c2) |
        //
        // det(M) = alpha^(s0*c1+s1*c2) XOR alpha^(s1*c1+s0*c2)
        //        = alpha^(s0*c1+s1*c2) * (1 XOR alpha^((s1-s0)*(c2-c1) mod 255))
        //
        // Non-zero iff (s1-s0)*(c2-c1) ≢ 0 (mod 255), i.e., neither factor
        // is 0 mod 255 (guaranteed by s0 != s1, c1 != c2, and small values).
        for s0 in 0u32..8 {
            for s1 in (s0+1)..8 {
                for c1 in 1u32..8 {
                    for c2 in (c1+1)..8 {
                        let a00 = Gf2_8::vandermonde(s0, c1);
                        let a01 = Gf2_8::vandermonde(s1, c1);
                        let a10 = Gf2_8::vandermonde(s0, c2);
                        let a11 = Gf2_8::vandermonde(s1, c2);
                        // det = a00*a11 + a01*a10  (XOR in char 2)
                        let det = a00.mul(a11).add(a01.mul(a10));
                        assert_ne!(det, Gf2_8::ZERO,
                            "Vandermonde 2×2 singular at s=({s0},{s1}) c=({c1},{c2})");
                    }
                }
            }
        }
    }
}