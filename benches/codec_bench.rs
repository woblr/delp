use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use bytes::Bytes;
use delp::{
    config::{EncoderConfig, DecoderConfig, Field},
    codec::{DefaultEncoder, DefaultDecoder, EncoderOutput, DecoderEvent},
    wire::{source::SourcePacket, coded::CodedPacket},
};

fn build_encoder(sym_size: usize, field: Field, fec_numer: usize, fec_denom: usize) -> DefaultEncoder {
    let cfg = EncoderConfig::builder(sym_size)
        .field(field)
        .window_capacity(256)
        .fec_rate(fec_numer, fec_denom)
        .build()
        .unwrap();
    DefaultEncoder::with_defaults(cfg)
}

fn bench_encoder_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("encoder_throughput");
    for sym_size in [256usize, 1024, 4096] {
        let payload = vec![0x42u8; sym_size];
        group.throughput(criterion::Throughput::Bytes(sym_size as u64));
        group.bench_with_input(
            BenchmarkId::new("gf2_8_1:4", sym_size),
            &sym_size,
            |b, _| {
                let mut enc = build_encoder(sym_size, Field::Gf2_8, 1, 4);
                b.iter(|| {
                    enc.submit_source(black_box(Bytes::copy_from_slice(&payload))).unwrap()
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("gf2_8_1:1", sym_size),
            &sym_size,
            |b, _| {
                let mut enc = build_encoder(sym_size, Field::Gf2_8, 1, 1);
                b.iter(|| {
                    enc.submit_source(black_box(Bytes::copy_from_slice(&payload))).unwrap()
                });
            },
        );
    }
    group.finish();
}

fn bench_encode_decode_roundtrip(c: &mut Criterion) {
    let sym_size = 1024;
    let n_syms   = 32;
    let payload  = vec![0xAAu8; sym_size];

    c.bench_function("encode_decode_no_loss_1024B_32sym", |b| {
        b.iter(|| {
            let enc_cfg = EncoderConfig::builder(sym_size)
                .window_capacity(64).fec_rate(1, 4).build().unwrap();
            let dec_cfg = DecoderConfig::builder(sym_size)
                .feedback_every(1000).build().unwrap();
            let mut enc = DefaultEncoder::with_defaults(enc_cfg);
            let mut dec = DefaultDecoder::with_defaults(dec_cfg);

            let mut delivered = 0usize;
            for _ in 0..n_syms {
                let out = enc.submit_source(Bytes::copy_from_slice(&payload)).unwrap();
                for item in out {
                    let evts = match item {
                        EncoderOutput::Source(raw) => {
                            let sp = SourcePacket::parse(&raw).unwrap();
                            dec.handle_source(&sp).unwrap()
                        }
                        EncoderOutput::Coded(raw) => {
                            let cp = CodedPacket::parse(&raw).unwrap();
                            dec.handle_coded(&cp).unwrap()
                        }
                    };
                    for ev in evts {
                        if matches!(ev, DecoderEvent::SourceReady { .. }) {
                            delivered += 1;
                        }
                    }
                }
            }
            black_box(delivered)
        });
    });
}

criterion_group!(benches, bench_encoder_throughput, bench_encode_decode_roundtrip);
criterion_main!(benches);