use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use delp::gf::simd::{mul_acc_gf2_8, mul_scale_gf2_8, mul_acc_gf2_4};

fn bench_mul_acc_gf2_8(c: &mut Criterion) {
    let mut group = c.benchmark_group("mul_acc_gf2_8");
    for size in [64usize, 256, 1024, 4096, 16384, 65536] {
        let src  = vec![0xABu8; size];
        let mut dst = vec![0u8; size];
        group.throughput(criterion::Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| mul_acc_gf2_8(black_box(&mut dst), black_box(&src), black_box(0x7F)));
        });
    }
    group.finish();
}

fn bench_mul_scale_gf2_8(c: &mut Criterion) {
    let mut group = c.benchmark_group("mul_scale_gf2_8");
    for size in [64usize, 1024, 16384, 65536] {
        let mut buf = vec![0xCDu8; size];
        group.throughput(criterion::Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| mul_scale_gf2_8(black_box(&mut buf), black_box(0x3A)));
        });
    }
    group.finish();
}

fn bench_mul_acc_gf2_4(c: &mut Criterion) {
    let mut group = c.benchmark_group("mul_acc_gf2_4");
    for size in [64usize, 1024, 16384] {
        let src  = vec![0x55u8; size];
        let mut dst = vec![0u8; size];
        group.throughput(criterion::Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| mul_acc_gf2_4(black_box(&mut dst), black_box(&src), black_box(0x37)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_mul_acc_gf2_8, bench_mul_scale_gf2_8, bench_mul_acc_gf2_4);
criterion_main!(benches);