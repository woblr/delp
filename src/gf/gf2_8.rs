use super::{Field, tables::{GF2_8_EXP, GF2_8_LOG, GF2_8_ORDER}};

/// An element of GF(2⁸).
///
/// The full byte range 0..=255 is used; each symbol payload byte is treated
/// as one GF(2^8) element.  This is the recommended field for most workloads.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(transparent)]
pub struct Gf2_8(u8);

impl Gf2_8 {
    #[inline(always)]
    pub const fn new(v: u8) -> Self { Self(v) }

    #[inline(always)]
    pub const fn value(self) -> u8 { self.0 }
}

#[inline(always)]
fn gf2_8_mul_raw(a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 { return 0; }
    // Doubled EXP table avoids `% 255` — just index directly.
    GF2_8_EXP[GF2_8_LOG[a as usize] as usize + GF2_8_LOG[b as usize] as usize]
}

impl Field for Gf2_8 {
    const ORDER: usize  = GF2_8_ORDER;
    const COEF_BITS: u8 = 8;
    const ZERO: Self    = Self(0);
    const ONE: Self     = Self(1);

    #[inline(always)]
    fn add(self, rhs: Self) -> Self { Self(self.0 ^ rhs.0) }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self { Self(gf2_8_mul_raw(self.0, rhs.0)) }

    #[inline(always)]
    fn inv(self) -> Self {
        // Inversion of zero is undefined in any field.  We assert in both
        // debug and release: a wrong inverse would silently corrupt all
        // downstream recovery, which is far harder to diagnose than a panic.
        assert_ne!(self.0, 0, "GF(2^8): inversion of the zero element is undefined");
        // inv(a) = alpha^(255 - log(a)); the doubled EXP table keeps the
        // index in [1..255] — LOG[0] = 0xFF guards the sentinel.
        let log_a = GF2_8_LOG[self.0 as usize] as usize;
        debug_assert_ne!(log_a, 0xFF, "LOG table sentinel leaked into inv path");
        Self(GF2_8_EXP[255 - log_a])
    }

    #[inline(always)]
    fn alpha_pow(exp: u32) -> Self {
        Self(GF2_8_EXP[(exp as usize) % 255])
    }

    #[inline(always)]
    fn from_u8(v: u8) -> Self { Self(v) }

    #[inline(always)]
    fn to_u8(self) -> u8 { self.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mul_spot() {
        let a = Gf2_8::alpha_pow(3);
        let b = Gf2_8::alpha_pow(7);
        assert_eq!(a.mul(b), Gf2_8::alpha_pow(10));
    }

    #[test]
    fn inv_all_nonzero() {
        for v in 1..=255u8 {
            let a = Gf2_8::new(v);
            assert_eq!(a.mul(a.inv()), Gf2_8::ONE);
        }
    }

    #[test]
    fn add_is_xor() {
        for a in 0..=255u8 {
            for b in 0..=255u8 {
                assert_eq!(Gf2_8::new(a).add(Gf2_8::new(b)), Gf2_8::new(a ^ b));
            }
        }
    }
}