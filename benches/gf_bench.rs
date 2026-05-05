//! GF(2^8) and GF(2^4) SIMD kernel benchmarks.
//!
//! Run with:
//!   cargo bench --bench gf_bench
//!   cargo bench --bench gf_bench -- mul_acc_gf2_8   # single group
//!
//! HTML report: target/criterion/

use criterion::{
    black_box, criterion_group, criterion_main,
    BenchmarkId, Criterion, Throughput,
};
use delp::gf::simd::ops::{mul_acc_gf2_8, mul_scale_gf2_8, mul_acc_gf2_4};
use delp::gf::simd::{mul_acc_gf2_8_reference, mul_scale_gf2_8_reference, mul_acc_gf2_4_reference};

// ── GF(2^8) mul-accumulate ────────────────────────────────────────────────

fn bench_mul_acc_gf2_8(c: &mut Criterion) {
    let mut group = c.benchmark_group("mul_acc_gf2_8");
    for size in [64usize, 256, 1024, 4096, 16384, 65536] {
        let src     = vec![0xABu8; size];
        let mut dst = vec![0u8; size];
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::new("simd", size), &size, |b, _| {
                b.iter(|| mul_acc_gf2_8(
                    black_box(&mut dst),
                    black_box(&src),
                    black_box(0x7F),
                ));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("scalar", size), &size, |b, _| {
                b.iter(|| mul_acc_gf2_8_reference(
                    black_box(&mut dst),
                    black_box(&src),
                    black_box(0x7F),
                ));
            },
        );
    }
    group.finish();
}

// ── GF(2^8) mul-scale (in-place) ─────────────────────────────────────────

fn bench_mul_scale_gf2_8(c: &mut Criterion) {
    let mut group = c.benchmark_group("mul_scale_gf2_8");
    for size in [64usize, 256, 1024, 4096, 16384, 65536] {
        let data = vec![0xCDu8; size];
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::new("simd", size), &size, |b, _| {
                let mut buf = data.clone();
                b.iter(|| mul_scale_gf2_8(black_box(&mut buf), black_box(0x3A)));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("scalar", size), &size, |b, _| {
                let mut buf = data.clone();
                b.iter(|| mul_scale_gf2_8_reference(black_box(&mut buf), black_box(0x3A)));
            },
        );
    }
    group.finish();
}

// ── GF(2^4) mul-accumulate (packed nibbles) ───────────────────────────────

fn bench_mul_acc_gf2_4(c: &mut Criterion) {
    let mut group = c.benchmark_group("mul_acc_gf2_4");
    for size in [64usize, 256, 1024, 4096, 16384] {
        let src     = vec![0x55u8; size];
        let mut dst = vec![0u8; size];
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::new("simd", size), &size, |b, _| {
                b.iter(|| mul_acc_gf2_4(
                    black_box(&mut dst),
                    black_box(&src),
                    black_box(0x37),
                ));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("scalar", size), &size, |b, _| {
                b.iter(|| mul_acc_gf2_4_reference(
                    black_box(&mut dst),
                    black_box(&src),
                    black_box(0x37),
                ));
            },
        );
    }
    group.finish();
}

// ── Coefficient sweep: all 256 coefs, fixed 1 KB buffer ──────────────────
//
// Measures dispatch overhead per coefficient value — useful to detect
// pathological behaviour for specific coefs (e.g. coef=0 early-return).

fn bench_coef_sweep_gf2_8(c: &mut Criterion) {
    let src     = vec![0x42u8; 1024];
    let mut dst = vec![0u8; 1024];
    c.bench_function("mul_acc_gf2_8/coef_sweep_1kb", |b| {
        b.iter(|| {
            for coef in 0u8..=255 {
                mul_acc_gf2_8(black_box(&mut dst), black_box(&src), black_box(coef));
            }
        });
    });
}

criterion_group!(
    benches,
    bench_mul_acc_gf2_8,
    bench_mul_scale_gf2_8,
    bench_mul_acc_gf2_4,
    bench_coef_sweep_gf2_8,
);
criterion_main!(benches);