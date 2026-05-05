/// SIMD-accelerated bulk GF arithmetic — `std::arch` + runtime dispatch.
///
/// Dispatch order (detected once at startup via `OnceLock`):
///
/// | Path   | Instruction     | Bytes/iter | Requires    |
/// |--------|-----------------|------------|-------------|
/// | AVX2   | vpshufb         | 32         | AVX2        |
/// | SSSE3  | pshufb          | 16         | SSSE3       |
/// | Scalar | nibble table    | 1          | always      |
///
/// **Algorithm — split-nibble table lookup:**
///
/// For coefficient `c`, precompute two 16-entry tables:
///   `lo[x] = c * x`       for x in 0..16  (low nibble products)
///   `hi[x] = c * (x<<4)`  for x in 0..16  (high nibble products)
///
/// Then for any byte `b`:
///   `c * b = lo[b & 0x0F] ^ hi[b >> 4]`
///
/// SSSE3 `pshufb` / AVX2 `vpshufb` perform 16/32 parallel table lookups
/// in a single instruction — the key to high throughput.
///
/// All 256 GF(2^8) nibble table pairs are precomputed in a static
/// `OnceLock` (8 KB), eliminating per-call table construction.

use std::sync::OnceLock;

use super::tables::{GF2_8_EXP, GF2_8_LOG, GF2_4_EXP, GF2_4_LOG, GF2_4_ORDER};

// ═══════════════════════════════════════════════════════════════
// Nibble lookup tables
// ═══════════════════════════════════════════════════════════════

/// Precomputed split-nibble tables for one GF coefficient.
pub struct NibbleTables {
    /// `lo[x] = coef * x`       for x in 0..16
    pub lo: [u8; 16],
    /// `hi[x] = coef * (x<<4)`  for x in 0..16
    pub hi: [u8; 16],
}

impl NibbleTables {
    fn for_gf2_8(coef: u8) -> Self {
        let mut lo = [0u8; 16];
        let mut hi = [0u8; 16];
        for x in 0u8..16 {
            lo[x as usize] = gf2_8_mul_raw(x, coef);
            hi[x as usize] = gf2_8_mul_raw(x << 4, coef);
        }
        NibbleTables { lo, hi }
    }

    /// `coef` is a packed byte: high nibble coef for high element,
    /// low nibble coef for low element.
    fn for_gf2_4(coef: u8) -> Self {
        let c_lo = coef & 0x0F;
        let c_hi = coef >> 4;
        let mut lo = [0u8; 16];
        let mut hi = [0u8; 16];
        for x in 0u8..16 {
            lo[x as usize] = gf2_4_mul_raw(x, c_lo);       // low element product
            hi[x as usize] = gf2_4_mul_raw(x, c_hi) << 4;  // high element product, pre-shifted
        }
        NibbleTables { lo, hi }
    }
}

// ═══════════════════════════════════════════════════════════════
// Precomputed static tables — initialised once on first use
// ═══════════════════════════════════════════════════════════════

static GF2_8_TBL: OnceLock<Box<[NibbleTables; 256]>> = OnceLock::new();
static GF2_4_TBL: OnceLock<Box<[NibbleTables; 256]>> = OnceLock::new();

#[inline]
fn gf2_8_tbl() -> &'static [NibbleTables; 256] {
    GF2_8_TBL.get_or_init(|| {
        Box::new(std::array::from_fn(|c| NibbleTables::for_gf2_8(c as u8)))
    })
}

#[inline]
fn gf2_4_tbl() -> &'static [NibbleTables; 256] {
    GF2_4_TBL.get_or_init(|| {
        Box::new(std::array::from_fn(|c| NibbleTables::for_gf2_4(c as u8)))
    })
}

// ═══════════════════════════════════════════════════════════════
// Runtime dispatch — detected once at startup
// ═══════════════════════════════════════════════════════════════

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dispatch { Avx2, Ssse3, Scalar }

static DISPATCH: OnceLock<Dispatch> = OnceLock::new();

#[inline]
fn dispatch() -> Dispatch {
    *DISPATCH.get_or_init(|| {
        if is_x86_feature_detected!("avx2")  { return Dispatch::Avx2; }
        if is_x86_feature_detected!("ssse3") { return Dispatch::Ssse3; }
        Dispatch::Scalar
    })
}

// ═══════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════

/// `dst[i] ^= coef * src[i]` over GF(2^8).
#[inline]
pub fn mul_acc_gf2_8(dst: &mut [u8], src: &[u8], coef: u8) {
    assert_eq!(dst.len(), src.len());
    if coef == 0 { return; }
    let t = &gf2_8_tbl()[coef as usize];
    match dispatch() {
        Dispatch::Avx2   => unsafe { avx2_mul_acc(dst, src, t) },
        Dispatch::Ssse3  => unsafe { ssse3_mul_acc(dst, src, t) },
        Dispatch::Scalar => scalar_mul_acc(dst, src, t),
    }
}

/// `buf[i] = coef * buf[i]` over GF(2^8), in-place.
#[inline]
pub fn mul_scale_gf2_8(buf: &mut [u8], coef: u8) {
    if coef == 0 { buf.fill(0); return; }
    if coef == 1 { return; }
    let t = &gf2_8_tbl()[coef as usize];
    match dispatch() {
        Dispatch::Avx2   => unsafe { avx2_mul_scale(buf, t) },
        Dispatch::Ssse3  => unsafe { ssse3_mul_scale(buf, t) },
        Dispatch::Scalar => scalar_mul_scale(buf, t),
    }
}

/// `dst[i] ^= packed_coef * src[i]` over GF(2^4), two elements per byte.
#[inline]
pub fn mul_acc_gf2_4(dst: &mut [u8], src: &[u8], coef: u8) {
    assert_eq!(dst.len(), src.len());
    if coef == 0 { return; }
    let t = &gf2_4_tbl()[coef as usize];
    match dispatch() {
        Dispatch::Avx2   => unsafe { avx2_mul_acc(dst, src, t) },
        Dispatch::Ssse3  => unsafe { ssse3_mul_acc(dst, src, t) },
        Dispatch::Scalar => scalar_mul_acc(dst, src, t),
    }
}

/// `buf[i] = packed_coef * buf[i]` over GF(2^4), in-place.
#[inline]
pub fn mul_scale_gf2_4(buf: &mut [u8], coef: u8) {
    if coef == 0 { buf.fill(0); return; }
    let t = &gf2_4_tbl()[coef as usize];
    match dispatch() {
        Dispatch::Avx2   => unsafe { avx2_mul_scale(buf, t) },
        Dispatch::Ssse3  => unsafe { ssse3_mul_scale(buf, t) },
        Dispatch::Scalar => scalar_mul_scale(buf, t),
    }
}

/// Re-export module for canonical import path.
pub mod ops {
    pub use super::{mul_acc_gf2_8, mul_scale_gf2_8, mul_acc_gf2_4, mul_scale_gf2_4};
}

// ═══════════════════════════════════════════════════════════════
// Scalar fallback
// ═══════════════════════════════════════════════════════════════

#[inline]
fn scalar_mul_acc(dst: &mut [u8], src: &[u8], t: &NibbleTables) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d ^= t.lo[(s & 0x0F) as usize] ^ t.hi[(s >> 4) as usize];
    }
}

#[inline]
fn scalar_mul_scale(dst: &mut [u8], t: &NibbleTables) {
    for d in dst.iter_mut() {
        let s = *d;
        *d = t.lo[(s & 0x0F) as usize] ^ t.hi[(s >> 4) as usize];
    }
}

// ═══════════════════════════════════════════════════════════════
// SSSE3 kernel — 16 bytes/iter
// ═══════════════════════════════════════════════════════════════

#[target_feature(enable = "ssse3")]
unsafe fn ssse3_mul_acc(dst: &mut [u8], src: &[u8], t: &NibbleTables) {
    use std::arch::x86_64::*;
    let lo_tbl  = _mm_loadu_si128(t.lo.as_ptr() as *const __m128i);
    let hi_tbl  = _mm_loadu_si128(t.hi.as_ptr() as *const __m128i);
    let mask_0f = _mm_set1_epi8(0x0F_u8 as i8);

    let chunks = dst.len() / 16;
    let tail   = dst.len() % 16;

    for i in 0..chunks {
        let off = i * 16;
        let inp = _mm_loadu_si128(src[off..].as_ptr() as *const __m128i);
        let lo_n = _mm_and_si128(inp, mask_0f);
        let hi_n = _mm_and_si128(_mm_srli_epi16(inp, 4), mask_0f);
        let prod = _mm_xor_si128(
            _mm_shuffle_epi8(lo_tbl, lo_n),
            _mm_shuffle_epi8(hi_tbl, hi_n),
        );
        let cur  = _mm_loadu_si128(dst[off..].as_ptr() as *const __m128i);
        _mm_storeu_si128(dst[off..].as_mut_ptr() as *mut __m128i,
                         _mm_xor_si128(cur, prod));
    }

    let base = chunks * 16;
    scalar_mul_acc(&mut dst[base..base + tail], &src[base..base + tail], t);
}

#[target_feature(enable = "ssse3")]
unsafe fn ssse3_mul_scale(dst: &mut [u8], t: &NibbleTables) {
    use std::arch::x86_64::*;
    let lo_tbl  = _mm_loadu_si128(t.lo.as_ptr() as *const __m128i);
    let hi_tbl  = _mm_loadu_si128(t.hi.as_ptr() as *const __m128i);
    let mask_0f = _mm_set1_epi8(0x0F_u8 as i8);

    let chunks = dst.len() / 16;
    let tail   = dst.len() % 16;

    for i in 0..chunks {
        let off = i * 16;
        let inp = _mm_loadu_si128(dst[off..].as_ptr() as *const __m128i);
        let lo_n = _mm_and_si128(inp, mask_0f);
        let hi_n = _mm_and_si128(_mm_srli_epi16(inp, 4), mask_0f);
        let prod = _mm_xor_si128(
            _mm_shuffle_epi8(lo_tbl, lo_n),
            _mm_shuffle_epi8(hi_tbl, hi_n),
        );
        _mm_storeu_si128(dst[off..].as_mut_ptr() as *mut __m128i, prod);
    }

    let base = chunks * 16;
    scalar_mul_scale(&mut dst[base..base + tail], t);
}

// ═══════════════════════════════════════════════════════════════
// AVX2 kernel — 32 bytes/iter (this machine: AVX2 detected)
// ═══════════════════════════════════════════════════════════════

#[target_feature(enable = "avx2")]
unsafe fn avx2_mul_acc(dst: &mut [u8], src: &[u8], t: &NibbleTables) {
    use std::arch::x86_64::*;
    let lo128 = _mm_loadu_si128(t.lo.as_ptr() as *const __m128i);
    let hi128 = _mm_loadu_si128(t.hi.as_ptr() as *const __m128i);
    let lo_tbl  = _mm256_broadcastsi128_si256(lo128);
    let hi_tbl  = _mm256_broadcastsi128_si256(hi128);
    let mask_0f = _mm256_set1_epi8(0x0F_u8 as i8);

    let chunks = dst.len() / 32;
    let rem    = dst.len() % 32;

    for i in 0..chunks {
        let off = i * 32;
        let inp = _mm256_loadu_si256(src[off..].as_ptr() as *const __m256i);
        let lo_n = _mm256_and_si256(inp, mask_0f);
        let hi_n = _mm256_and_si256(_mm256_srli_epi16(inp, 4), mask_0f);
        let prod = _mm256_xor_si256(
            _mm256_shuffle_epi8(lo_tbl, lo_n),
            _mm256_shuffle_epi8(hi_tbl, hi_n),
        );
        let cur  = _mm256_loadu_si256(dst[off..].as_ptr() as *const __m256i);
        _mm256_storeu_si256(dst[off..].as_mut_ptr() as *mut __m256i,
                            _mm256_xor_si256(cur, prod));
    }

    let base = chunks * 32;
    // Remainder handled by SSSE3 for up to 16 bytes, then scalar tail
    ssse3_mul_acc(&mut dst[base..base + rem], &src[base..base + rem], t);
}

#[target_feature(enable = "avx2")]
unsafe fn avx2_mul_scale(dst: &mut [u8], t: &NibbleTables) {
    use std::arch::x86_64::*;
    let lo128 = _mm_loadu_si128(t.lo.as_ptr() as *const __m128i);
    let hi128 = _mm_loadu_si128(t.hi.as_ptr() as *const __m128i);
    let lo_tbl  = _mm256_broadcastsi128_si256(lo128);
    let hi_tbl  = _mm256_broadcastsi128_si256(hi128);
    let mask_0f = _mm256_set1_epi8(0x0F_u8 as i8);

    let chunks = dst.len() / 32;
    let rem    = dst.len() % 32;

    for i in 0..chunks {
        let off = i * 32;
        let inp = _mm256_loadu_si256(dst[off..].as_ptr() as *const __m256i);
        let lo_n = _mm256_and_si256(inp, mask_0f);
        let hi_n = _mm256_and_si256(_mm256_srli_epi16(inp, 4), mask_0f);
        let prod = _mm256_xor_si256(
            _mm256_shuffle_epi8(lo_tbl, lo_n),
            _mm256_shuffle_epi8(hi_tbl, hi_n),
        );
        _mm256_storeu_si256(dst[off..].as_mut_ptr() as *mut __m256i, prod);
    }

    let base = chunks * 32;
    ssse3_mul_scale(&mut dst[base..base + rem], t);
}

// ═══════════════════════════════════════════════════════════════
// Raw GF scalar multiply (for table construction only)
// ═══════════════════════════════════════════════════════════════

#[inline(always)]
fn gf2_8_mul_raw(a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 { return 0; }
    GF2_8_EXP[GF2_8_LOG[a as usize] as usize + GF2_8_LOG[b as usize] as usize]
}

#[inline(always)]
fn gf2_4_mul_raw(a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 { return 0; }
    GF2_4_EXP[(GF2_4_LOG[a as usize] as usize + GF2_4_LOG[b as usize] as usize)
        % (GF2_4_ORDER - 1)]
}


// ═══════════════════════════════════════════════════════════════
// Test oracles
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
pub fn mul_acc_gf2_8_reference(dst: &mut [u8], src: &[u8], coef: u8) {
    for (d, &s) in dst.iter_mut().zip(src) { *d ^= gf2_8_mul_raw(s, coef); }
}
#[cfg(test)]
pub fn mul_scale_gf2_8_reference(buf: &mut [u8], coef: u8) {
    for b in buf.iter_mut() { *b = gf2_8_mul_raw(*b, coef); }
}
#[cfg(test)]
pub fn mul_acc_gf2_4_reference(dst: &mut [u8], src: &[u8], coef: u8) {
    for (d, &s) in dst.iter_mut().zip(src) { *d ^= gf2_4_mul_packed_scalar(s, coef); }
}

#[cfg(test)]
mod tests {
    use super::ops::*;
    use super::{mul_acc_gf2_8_reference, mul_scale_gf2_8_reference,
                mul_acc_gf2_4_reference, dispatch, Dispatch};

    fn payload(len: usize) -> Vec<u8> {
        (0..len).map(|i| ((i * 7 + 13) & 0xFF) as u8).collect()
    }

    #[test]
    fn dispatch_is_avx2_on_this_machine() {
        assert_eq!(dispatch(), Dispatch::Avx2);
    }

    #[test]
    fn gf2_8_acc_all_coefficients() {
        for coef in 0u8..=255 {
            let src = payload(256);
            let mut s = vec![0u8; 256];
            let mut r = vec![0u8; 256];
            mul_acc_gf2_8(&mut s, &src, coef);
            mul_acc_gf2_8_reference(&mut r, &src, coef);
            assert_eq!(s, r, "coef={coef}");
        }
    }

    #[test]
    fn gf2_8_scale_all_coefficients() {
        for coef in 0u8..=255 {
            let data = payload(256);
            let mut s = data.clone();
            let mut r = data.clone();
            mul_scale_gf2_8(&mut s, coef);
            mul_scale_gf2_8_reference(&mut r, coef);
            assert_eq!(s, r, "coef={coef}");
        }
    }

    #[test]
    fn gf2_4_acc_all_coefficients() {
        for coef in 0u8..=255 {
            let src = payload(256);
            let mut s = vec![0u8; 256];
            let mut r = vec![0u8; 256];
            mul_acc_gf2_4(&mut s, &src, coef);
            mul_acc_gf2_4_reference(&mut r, &src, coef);
            assert_eq!(s, r, "coef={coef}");
        }
    }

    #[test]
    fn tail_bytes_all_lengths_1_to_63() {
        let src: Vec<u8> = (0..64).collect();
        for len in 1..=63 {
            let mut a = vec![0u8; len];
            let mut b = vec![0u8; len];
            mul_acc_gf2_8(&mut a, &src[..len], 0xAB);
            mul_acc_gf2_8_reference(&mut b, &src[..len], 0xAB);
            assert_eq!(a, b, "len={len}");
        }
    }

    #[test]
    fn avx2_chunks_and_tail() {
        // 33 bytes = 1×32 AVX2 chunk + 1 byte tail
        let src = payload(33);
        let mut a = vec![0u8; 33];
        let mut b = vec![0u8; 33];
        mul_acc_gf2_8(&mut a, &src, 0x7F);
        mul_acc_gf2_8_reference(&mut b, &src, 0x7F);
        assert_eq!(a, b);
    }

    #[test]
    fn coef_zero_noop() {
        let src = payload(64);
        let mut dst = vec![0xABu8; 64];
        let orig = dst.clone();
        mul_acc_gf2_8(&mut dst, &src, 0);
        assert_eq!(dst, orig);
    }

    #[test]
    fn coef_zero_scale_zeros() {
        let mut buf = payload(64);
        mul_scale_gf2_8(&mut buf, 0);
        assert!(buf.iter().all(|&b| b == 0));
    }
}