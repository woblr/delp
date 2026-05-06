//! Encoder pipeline benchmarks.
//!
//! Run with:
//!   cargo bench --bench codec_bench
//!
//! Measures:
//!   - Full encode pipeline: submit_source × N + generate_coded
//!   - Vandermonde vs Cauchy coefficient strategies
//!   - symbol_size × window_size matrix at realistic MTU sizes

use bytes::Bytes;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use delp::codec::encoder::DefaultEncoder;
use delp::config::{EncoderConfig, Field, MatrixStrategy};

// ── Helper: build a pre-loaded encoder ───────────────────────────────────

fn make_loaded_encoder(sym_size: usize, n_src: usize, strategy: MatrixStrategy) -> DefaultEncoder {
    let cap = match strategy {
        MatrixStrategy::Cauchy => n_src.min(128),
        MatrixStrategy::Vandermonde => n_src,
    };
    let cfg = EncoderConfig::builder(sym_size)
        .window_capacity(cap)
        .fec_rate(0, 1) // FEC disabled during submit — we call generate_coded manually
        .matrix_strategy(strategy)
        .build()
        .unwrap();
    let mut enc = DefaultEncoder::with_defaults(cfg);
    let payload = vec![0xABu8; sym_size];
    for _ in 0..n_src {
        enc.submit_source(Bytes::from(payload.clone())).unwrap();
    }
    enc
}

// ── Benchmark: generate_coded throughput ─────────────────────────────────
//
// Measures how quickly one coded packet is produced from a window of N
// symbols at various symbol sizes.  The encoder is reset each iteration
// via `iter_with_setup` to avoid coded_id exhaustion during the bench.

fn bench_generate_coded(c: &mut Criterion) {
    let mut group = c.benchmark_group("generate_coded");

    for (sym_size, n_src) in [(64, 16), (512, 16), (1500, 8), (4096, 4)] {
        let throughput_bytes = (sym_size * n_src) as u64;
        group.throughput(Throughput::Bytes(throughput_bytes));

        // Vandermonde
        group.bench_with_input(
            BenchmarkId::new("vandermonde", format!("{sym_size}B×{n_src}")),
            &(sym_size, n_src),
            |b, &(sz, n)| {
                b.iter_with_setup(
                    || make_loaded_encoder(sz, n, MatrixStrategy::Vandermonde),
                    |mut enc| black_box(enc.generate_coded().unwrap()),
                );
            },
        );

        // Cauchy (only if n_src <= 128 — enforced by builder)
        if n_src <= 128 {
            group.bench_with_input(
                BenchmarkId::new("cauchy", format!("{sym_size}B×{n_src}")),
                &(sym_size, n_src),
                |b, &(sz, n)| {
                    b.iter_with_setup(
                        || make_loaded_encoder(sz, n, MatrixStrategy::Cauchy),
                        |mut enc| black_box(enc.generate_coded().unwrap()),
                    );
                },
            );
        }
    }
    group.finish();
}

// ── Benchmark: submit_source end-to-end ──────────────────────────────────
//
// Measures a full N-symbol encode session with 1:1 FEC rate.
// Each call to submit_source emits one source packet and one coded packet.

fn bench_submit_source_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("submit_source_pipeline");

    for sym_size in [64usize, 512, 1500] {
        let n_src = 8usize;
        group.throughput(Throughput::Bytes((sym_size * n_src) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{sym_size}B×{n_src}")),
            &(sym_size, n_src),
            |b, &(sz, n)| {
                let cfg = EncoderConfig::builder(sz)
                    .window_capacity(n)
                    .fec_rate(1, 1)
                    .build()
                    .unwrap();
                let payload = vec![0xABu8; sz];
                b.iter(|| {
                    let mut enc = DefaultEncoder::with_defaults(cfg.clone());
                    for _ in 0..n {
                        black_box(enc.submit_source(Bytes::from(payload.clone())).unwrap());
                    }
                });
            },
        );
    }
    group.finish();
}

// ── Benchmark: GF(2^4) vs GF(2^8) at small symbol sizes ─────────────────

fn bench_field_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("field_comparison");
    let sym_size = 64usize;
    let n_src = 8usize;

    group.throughput(Throughput::Bytes((sym_size * n_src) as u64));

    for (label, field) in [("gf2_8", Field::Gf2_8), ("gf2_4", Field::Gf2_4)] {
        group.bench_function(label, |b| {
            b.iter_with_setup(
                || {
                    let cfg = EncoderConfig::builder(sym_size)
                        .window_capacity(n_src)
                        .fec_rate(0, 1)
                        .field(field)
                        .build()
                        .unwrap();
                    let mut enc = DefaultEncoder::with_defaults(cfg);
                    let payload = vec![0x42u8; sym_size];
                    for _ in 0..n_src {
                        enc.submit_source(Bytes::from(payload.clone())).unwrap();
                    }
                    enc
                },
                |mut enc| black_box(enc.generate_coded().unwrap()),
            );
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_generate_coded,
    bench_submit_source_pipeline,
    bench_field_comparison,
);
criterion_main!(benches);
