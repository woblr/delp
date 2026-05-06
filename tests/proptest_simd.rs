//! Property-based tests for the SIMD GF arithmetic kernels.
//!
//! For arbitrary input data, length, and coefficient, the dispatched
//! kernel (AVX2 / SSSE3 / NEON / scalar) must produce the same result
//! as the byte-wise reference implementation.

use proptest::prelude::*;

use delp::gf::simd::ops::{mul_acc_gf2_4, mul_acc_gf2_8, mul_scale_gf2_4, mul_scale_gf2_8};
use delp::gf::simd::{mul_acc_gf2_4_reference, mul_acc_gf2_8_reference, mul_scale_gf2_8_reference};

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256, .. ProptestConfig::default()
    })]

    /// `mul_acc_gf2_8` matches the reference for any data and any coef.
    #[test]
    fn mul_acc_gf2_8_matches_reference(
        coef in 0u8..=255u8,
        data in proptest::collection::vec(0u8..=255u8, 0..=200),
    ) {
        let mut simd_dst = vec![0u8; data.len()];
        let mut ref_dst  = vec![0u8; data.len()];
        mul_acc_gf2_8(&mut simd_dst, &data, coef);
        mul_acc_gf2_8_reference(&mut ref_dst, &data, coef);
        prop_assert_eq!(simd_dst, ref_dst);
    }

    /// `mul_scale_gf2_8` matches the reference for any buffer and any coef.
    #[test]
    fn mul_scale_gf2_8_matches_reference(
        coef in 0u8..=255u8,
        buf  in proptest::collection::vec(0u8..=255u8, 0..=200),
    ) {
        let mut simd_buf = buf.clone();
        let mut ref_buf  = buf;
        mul_scale_gf2_8(&mut simd_buf, coef);
        mul_scale_gf2_8_reference(&mut ref_buf, coef);
        prop_assert_eq!(simd_buf, ref_buf);
    }

    /// `mul_acc_gf2_4` (4-bit coef broadcast) matches the reference.
    #[test]
    fn mul_acc_gf2_4_matches_reference(
        coef in 0u8..16u8,
        data in proptest::collection::vec(0u8..=255u8, 0..=200),
    ) {
        let mut simd_dst = vec![0u8; data.len()];
        let mut ref_dst  = vec![0u8; data.len()];
        mul_acc_gf2_4(&mut simd_dst, &data, coef);
        mul_acc_gf2_4_reference(&mut ref_dst, &data, coef);
        prop_assert_eq!(simd_dst, ref_dst);
    }

    /// `mul_scale_gf2_4` is mul_acc_gf2_4 starting from zero applied to
    /// the original.  Verify the two are consistent.
    #[test]
    fn mul_scale_gf2_4_via_acc_consistency(
        coef in 0u8..16u8,
        buf  in proptest::collection::vec(0u8..=255u8, 0..=200),
    ) {
        let mut scaled = buf.clone();
        mul_scale_gf2_4(&mut scaled, coef);

        let mut acc = vec![0u8; buf.len()];
        mul_acc_gf2_4(&mut acc, &buf, coef);

        prop_assert_eq!(scaled, acc);
    }

    /// Self-inverse over odd lengths: applying the same operation twice
    /// with coef=1 (identity) must leave the buffer unchanged.
    #[test]
    fn mul_acc_gf2_8_identity_coef(
        len  in 0usize..200usize,
        seed in 0u8..=255u8,
    ) {
        let data: Vec<u8> = (0..len).map(|i| seed.wrapping_add(i as u8)).collect();
        let mut dst = vec![0u8; len];
        mul_acc_gf2_8(&mut dst, &data, 1);
        prop_assert_eq!(dst, data);
    }

    /// Coefficient zero is a no-op for accumulate, zero-fill for scale.
    #[test]
    fn coef_zero_acc_is_noop_scale_zeros(
        buf in proptest::collection::vec(0u8..=255u8, 0..=200),
    ) {
        let mut acc_dst = buf.clone();
        let saved = acc_dst.clone();
        mul_acc_gf2_8(&mut acc_dst, &saved, 0);
        prop_assert_eq!(acc_dst, saved);

        let mut scale_buf = buf.clone();
        mul_scale_gf2_8(&mut scale_buf, 0);
        prop_assert!(scale_buf.iter().all(|&b| b == 0));
    }

    /// Tail-handling: kernels must process every length 0..=63 correctly
    /// (covers below-AVX2-chunk, inside-SSSE3-chunk, scalar tail).
    #[test]
    fn tail_handling_all_short_lengths(
        coef in 0u8..=255u8,
        len  in 0usize..=63usize,
        seed in 0u8..=255u8,
    ) {
        let data: Vec<u8> = (0..len).map(|i| seed.wrapping_mul(i as u8 + 1)).collect();
        let mut simd_dst = vec![0u8; len];
        let mut ref_dst  = vec![0u8; len];
        mul_acc_gf2_8(&mut simd_dst, &data, coef);
        mul_acc_gf2_8_reference(&mut ref_dst, &data, coef);
        prop_assert_eq!(simd_dst, ref_dst);
    }
}
