use super::{
    tables::{GF2_4_EXP, GF2_4_LOG, GF2_4_ORDER},
    Field,
};

/// An element of GF(2⁴).
///
/// The value is stored in the low 4 bits of a `u8`; the high nibble is
/// always zero.  Two `Gf2_4` elements are packed into each byte of a
/// symbol payload (high nibble = element at even index, low nibble = odd).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(transparent)]
pub struct Gf2_4(u8);

impl Gf2_4 {
    /// Construct directly from a nibble value (0..=15).
    #[inline(always)]
    pub const fn new(v: u8) -> Self {
        debug_assert!(v < 16, "Gf2_4 value must be < 16");
        Self(v & 0x0F)
    }

    /// Raw nibble value.
    #[inline(always)]
    pub const fn value(self) -> u8 {
        self.0
    }

    // ── Packed-byte helpers ──────────────────────────────────────────────

    /// Multiply two packed bytes (each byte = two GF(2^4) elements,
    /// high nibble × high nibble, low nibble × low nibble) by a scalar.
    ///
    /// Used in the encoder hot path instead of element-by-element iteration.
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn mul_packed_byte(byte: u8, coef: u8) -> u8 {
        let hi_in = (byte >> 4) & 0x0F;
        let lo_in = byte & 0x0F;
        let c_hi = (coef >> 4) & 0x0F;
        let c_lo = coef & 0x0F;
        let hi_out = gf2_4_mul_raw(hi_in, c_hi);
        let lo_out = gf2_4_mul_raw(lo_in, c_lo);
        (hi_out << 4) | lo_out
    }

    /// XOR-accumulate `coef * src_byte` into `dst_byte` in GF(2^4) packed form.
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn mul_acc_byte(dst: u8, src: u8, coef: u8) -> u8 {
        dst ^ Self::mul_packed_byte(src, coef)
    }
}

/// Scalar multiply two raw nibble values (0..15) in GF(2^4).
#[inline(always)]
fn gf2_4_mul_raw(a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 {
        return 0;
    }
    let sum = GF2_4_LOG[a as usize] as usize + GF2_4_LOG[b as usize] as usize;
    GF2_4_EXP[sum % (GF2_4_ORDER - 1)]
}

impl Field for Gf2_4 {
    const ORDER: usize = GF2_4_ORDER;
    const COEF_BITS: u8 = 4;
    const ZERO: Self = Self(0);
    const ONE: Self = Self(1);

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(gf2_4_mul_raw(self.0, rhs.0))
    }

    #[inline(always)]
    fn inv(self) -> Self {
        // Same rationale as GF(2^8): assert in both debug and release.
        assert_ne!(
            self.0, 0,
            "GF(2^4): inversion of the zero element is undefined"
        );
        // inv(a) = alpha^(15 - log(a));  LOG[0] = 0xFF (sentinel — never reached here)
        let log_a = GF2_4_LOG[self.0 as usize] as usize;
        debug_assert_ne!(log_a, 0xFF, "GF2_4 LOG sentinel leaked into inv path");
        Self(GF2_4_EXP[(GF2_4_ORDER - 1 - log_a) % (GF2_4_ORDER - 1)])
    }

    #[inline(always)]
    fn alpha_pow(exp: u32) -> Self {
        Self(GF2_4_EXP[(exp as usize) % (GF2_4_ORDER - 1)])
    }

    #[inline(always)]
    fn from_u8(v: u8) -> Self {
        Self(v & 0x0F)
    }

    #[inline(always)]
    fn to_u8(self) -> u8 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mul_table_spot_checks() {
        // alpha = 2;  alpha^2 = 4;  alpha^2 * alpha^3 = alpha^5
        let a = Gf2_4::alpha_pow(2);
        let b = Gf2_4::alpha_pow(3);
        let c = Gf2_4::alpha_pow(5);
        assert_eq!(a.mul(b), c);
    }

    #[test]
    fn packed_byte_roundtrip() {
        // Multiply by ONE should be identity
        for byte in 0u8..=255 {
            assert_eq!(Gf2_4::mul_packed_byte(byte, 0x11), byte);
        }
    }

    #[test]
    fn packed_by_zero_is_zero() {
        for byte in 0u8..=255 {
            assert_eq!(Gf2_4::mul_packed_byte(byte, 0x00), 0);
        }
    }
}
