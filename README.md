# delp

Production-grade pure-Rust forward error correction (FEC) library for lossy
networks.  Implements an elastic-window erasure code: the sender XORs source
packets into coded (repair) packets using Galois field arithmetic; the
receiver recovers lost packets by solving a linear system over the same
field.  The encoding window adapts in real time based on receiver feedback.

```toml
[dependencies]
delp = "1"

# Async UDP transport (FecSender, FecReceiver, DelpSession):
delp = { version = "1", features = ["async"] }
```

## Why

Standard block codes commit to a code rate up front and incur full decoder
latency before any source symbol becomes available.  Sliding-window codes
(RFC 9407 / Tetrys) emit each source packet immediately *and* a small
fraction of coded packets; the decoder recovers losses on the fly without
retransmissions.  This keeps tail latency low on lossy links — wireless,
satellite, real-time media — where retransmits are expensive or impossible.

## Features

- **Pure state machine** — codec has no I/O, no async runtime, no threads.
  All methods are synchronous `&mut self` transitions; transport is the
  caller's concern.
- **Async UDP transport** — optional `async` feature ships
  `FecSender` / `FecReceiver` / `DelpSession` over `tokio::net::UdpSocket`,
  including `futures_core::Stream` and `futures_sink::Sink` impls.
- **GF(2⁴) and GF(2⁸)** finite fields with compile-time log/exp tables
  (zero startup cost).
- **SIMD-accelerated bulk arithmetic** — AVX2 (64 B/iter, 2× unrolled) /
  SSSE3 (16 B/iter) / aarch64 NEON (16 B/iter via `vqtbl1q_u8`) /
  scalar fallback, dispatched once at startup.
- **Two MDS-grade matrix strategies**: Vandermonde (cheap to compute) and
  Cauchy (mathematically proven full-rank submatrices for any erasure
  pattern).
- **Pluggable policies** for window eviction, congestion control, FEC
  rate, and feedback scheduling — swappable without forking the codec.
- **Multi-receiver ACK tracking** with cumulative + selective ACK
  (SACK) bit-vector encoding.
- **Zero-copy wire format** — packets parse via `zerocopy`; payloads
  are `bytes::Bytes` slices on the hot path.
- **Backpressure flow control** — `BackpressureMode::Reject` returns
  `Err(WindowFull)` instead of evicting unacknowledged symbols.

### What's novel — beyond RFC 9407 / Tetrys

Two delp-specific extensions that other sliding-window FEC libraries don't
implement:

- **Unlimited-length sessions via Generation Rotation.** Standard Cauchy
  GF(2⁸) coding maxes out at 128 coded packets per session: there are only
  128 disjoint y-points.  delp adds a 1-byte `generation` field to the
  encoding vector that rotates the y-point assignment by `(coded_id +
  gen·K) mod cycle` (`K` coprime to `cycle`).  Each generation yields 128
  fresh, linearly-independent coded packets; with a `u8` counter that's
  32 768 packets per session.  The wire change is backward compatible —
  `generation = 0` is bit-identical to RFC 9407.
- **Adaptive Loss-Targeted Coding (ALTC).** RFC 9407 always covers the
  *entire* window in every coded packet, even when most of those symbols
  have already been delivered.  delp lets the encoder generate a coded
  packet over an arbitrary subset of the window:
  - `generate_coded_targeted(&[id])` — explicit subset
  - `generate_coded_recent(n)` — most-recent `n` symbols (recent-loss
    prioritisation)
  - `generate_coded_for_receiver(rid)` — only symbols not yet ACK'd by a
    specific receiver, computed from the per-receiver SACK state
  Smaller cover sets translate directly to fewer GF multiplications at
  the encoder and sparser matrix rows at the decoder.  Cauchy coded
  packets also shrink linearly on the wire because their explicit
  coefficient list contracts with the cover.

## Architecture

```text
┌────────────────────────────────────────────────────┐
│  config      EncoderConfig / DecoderConfig         │
│  error       DelpError enum                        │
├────────────────────────────────────────────────────┤
│  gf/         GF(2⁴) + GF(2⁸) + SIMD mul_acc        │
├────────────────────────────────────────────────────┤
│  wire/       binary packet wire format             │
│    common    CommonHeader                          │
│    source    SourcePacket                          │
│    coded     CodedPacket                           │
│    feedback  FeedbackPacket (window update + SACK) │
│    ev/       EncodingVector + 4 ID storage formats │
├────────────────────────────────────────────────────┤
│  policy/     Pluggable strategy traits             │
│    WindowPolicy       AnyAck / AllAck / Quorum     │
│    CongestionControl  NoCC                         │
│    FecRateController  Constant / Adaptive          │
│    FeedbackPolicy     Constant / Immediate         │
├────────────────────────────────────────────────────┤
│  codec/      Pure state machines (no I/O)          │
│    encoder   Encoder<W,C,F> + sliding window       │
│    decoder   Decoder<P>     + matrix + buffer      │
├────────────────────────────────────────────────────┤
│  transport/  Async UDP layer (feature = "async")   │
│    FecSender / FecReceiver / DelpSession           │
└────────────────────────────────────────────────────┘
```

## Quick start — sync codec

```rust
use bytes::Bytes;
use delp::{
    config::{EncoderConfig, DecoderConfig},
    codec::{DefaultEncoder, DefaultDecoder, EncoderOutput, DecoderEvent},
    wire::{source::SourcePacket, coded::CodedPacket},
};

let enc_cfg = EncoderConfig::builder(1024)
    .window_capacity(32)
    .fec_rate(1, 4)             // 25 % redundancy
    .build()?;
let dec_cfg = DecoderConfig::builder(1024).build()?;

let mut enc = DefaultEncoder::with_defaults(enc_cfg);
let mut dec = DefaultDecoder::with_defaults(dec_cfg);

// Submit one source symbol → encoder emits 1 source + 0..N coded packets
let symbol = Bytes::from(vec![0x42u8; 1024]);
for pkt in enc.submit_source(symbol)? {
    match pkt {
        EncoderOutput::Source(raw) => {
            let sp = SourcePacket::parse(&raw)?;
            for ev in dec.handle_source(&sp)? {
                if let DecoderEvent::SourceReady { id, data } = ev {
                    println!("delivered {id}: {} B", data.len());
                }
            }
        }
        EncoderOutput::Coded(raw) => {
            let cp = CodedPacket::parse(&raw)?;
            let _ = dec.handle_coded(&cp)?;
        }
    }
}
```

## Quick start — async UDP

```rust
use delp::transport::session::SessionBuilder;

let session = SessionBuilder::new()
    .symbol_size(1024)
    .window_capacity(32)
    .fec_rate(1, 2)
    .build("0.0.0.0:5000".parse()?, "192.168.1.2:5001".parse()?)
    .await?;

let (mut tx, mut rx) = session.split();
tokio::spawn(async move { tx.send_source(payload).await });
tokio::spawn(async move {
    while let Ok((id, data)) = rx.recv_source().await {
        // ...
    }
});
```

## Examples

- [`udp_fec.rs`](examples/udp_fec.rs) — minimal end-to-end FEC over a
  loopback UDP link with simulated 30 % source-packet loss.
- [`file_transfer.rs`](examples/file_transfer.rs) — chunks a 128 KB blob
  into symbols, transfers it through a lossy proxy, and verifies
  byte-exact reconstruction on the other end.

```bash
cargo run --example udp_fec        --features async
cargo run --example file_transfer  --features async --release
```

## Matrix strategies

| Strategy    | Field   | Cycle length | Effective per session¹ | MDS guarantee |
|-------------|---------|--------------|------------------------|---------------|
| Vandermonde | GF(2⁸)  | 254          | unlimited (sliding)    | empirical     |
| Vandermonde | GF(2⁴)  | 14           | unlimited (sliding)    | empirical     |
| Cauchy      | GF(2⁸)  | 128          | **128 × 256 = 32 768** | **proven**    |
| Cauchy      | GF(2⁴)  | 7            | 7 × 256 = 1 792        | **proven**    |

¹ With the new `generation`-rotation extension.  Cauchy bumps a `u8`
generation counter every full coded-id cycle, rotating the y-point
assignment so successive cycles produce linearly-independent rows.
Vandermonde relies on the sliding window: as the window advances the
same coded-id label, paired with new source IDs in the EV, naturally
yields fresh equations.

When you need guaranteed recovery from any k-erasure pattern, choose
`MatrixStrategy::Cauchy`; otherwise the default `Vandermonde` is
slightly cheaper to compute.

## Performance

Criterion benchmarks ship under `benches/`:

```bash
cargo bench
```

On x86_64 with AVX2 the SIMD `mul_acc_gf2_8` kernel processes 64 bytes
per loop iteration (2× unrolled, two independent dependency chains).
Coefficient generation, dispatch, and nibble tables are all hoisted out
of the inner loop in the encoder's `compute_coded_payload`.

## Testing

```bash
cargo test --all-features
```

136+ tests across:
- 98 unit tests — field axioms, exhaustive small GF tables, wire-format
  round-trips, policy semantics
- 8 ALTC integration tests — targeted coding, per-receiver coverage,
  recovery correctness
- 5 async UDP transport integration tests
- 3 long-session Cauchy tests (1 200-packet sessions, generation
  rotation through 9+ cycles)
- 8 stress tests — random / burst / head loss patterns
- 5 codec property tests + 7 SIMD property tests (proptest)
- 2 doc tests

CI (`.github/workflows/ci.yml`) runs `cargo fmt --check`, `clippy
-D warnings`, the full test suite under `--no-default-features` /
default / `--all-features`, an aarch64 cross-compile, an MSRV build,
`cargo audit`, and `cargo doc -D warnings`.

## License

Apache-2.0
