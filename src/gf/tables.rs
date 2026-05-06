//! Compile-time log/exp table generation for GF(2^n).
//!
//! All functions are `const fn` so the tables live in the binary's `.rodata`
//! section — no runtime heap allocation, no `OnceLock`.

// ── GF(2^4) — irreducible poly x^4 + x + 1 = 0b10011 = 19 ──────────────

pub const GF2_4_ORDER: usize = 16;
pub const GF2_4_POLY: u8 = 0b0001_0011; // x^4 + x + 1, primitive poly

pub const GF2_4_EXP: [u8; GF2_4_ORDER] = {
    let mut exp = [0u8; GF2_4_ORDER];
    let mut x: u8 = 1;
    let mut i = 0usize;
    while i < GF2_4_ORDER - 1 {
        exp[i] = x;
        x <<= 1;
        if x & 0x10 != 0 {
            x ^= GF2_4_POLY & 0x0F;
            // reduce mod the irreducible poly: only keep low 4 bits
            // and XOR with x + 1 (poly without leading term)
            x ^= 0x10;
            x &= 0x0F;
        }
        i += 1;
    }
    // Last entry unused but fill for completeness
    exp[GF2_4_ORDER - 1] = 1; // alpha^15 == 1 (primitive)
    exp
};

/// LOG[0] is undefined; we store 0xFF as a sentinel — never read it for
/// division/inversion of zero (protected by debug_assert in the trait).
pub const GF2_4_LOG: [u8; GF2_4_ORDER] = {
    let mut log = [0xFFu8; GF2_4_ORDER];
    let mut i = 0usize;
    while i < GF2_4_ORDER - 1 {
        log[GF2_4_EXP[i] as usize] = i as u8;
        i += 1;
    }
    log
};

// ── GF(2^8) — irreducible poly x^8+x^4+x^3+x^2+1 = 0x11D ───────────────

pub const GF2_8_ORDER: usize = 256;
pub const GF2_8_POLY: u16 = 0x11D; // x^8 + x^4 + x^3 + x^2 + 1

/// EXP table is doubled (512 entries) to avoid modular index arithmetic
/// in the hot multiplication path: `EXP[(LOG[a] + LOG[b])]` instead of
/// `EXP[(LOG[a] + LOG[b]) % 255]`.
pub const GF2_8_EXP: [u8; 512] = {
    let mut exp = [0u8; 512];
    let mut x: u16 = 1;
    let mut i = 0usize;
    while i < 255 {
        exp[i] = x as u8;
        x <<= 1;
        if x & 0x100 != 0 {
            x ^= GF2_8_POLY;
        }
        i += 1;
    }
    // Mirror into upper half so we never need `% 255`
    let mut j = 0usize;
    while j < 255 {
        exp[255 + j] = exp[j];
        j += 1;
    }
    exp[510] = exp[0]; // one extra for the edge case
    exp
};

/// LOG[0] = 0xFF — explicit sentinel for "undefined" (log of zero).
/// The multiplication and inversion paths always guard with an early
/// zero-check BEFORE indexing this table, so the sentinel is never
/// dereferenced in correct code.  It also makes silent misuse
/// detectable: any stray read of LOG[0] produces 0xFF, which will
/// produce a nonsensical EXP index and a wrong (but obviously wrong)
/// result rather than a quietly plausible one.
pub const GF2_8_LOG: [u8; GF2_8_ORDER] = {
    let mut log = [0xFFu8; GF2_8_ORDER]; // 0xFF = "undefined" for index 0
    let mut i = 0usize;
    while i < 255 {
        log[GF2_8_EXP[i] as usize] = i as u8;
        i += 1;
    }
    log
};
