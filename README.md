# delp

Forward error correction for lossy networks, written in Rust.

The encoder keeps a sliding window of source packets and emits redundant "coded" packets that are linear combinations of recent symbols over a Galois field. When a source packet is lost, the receiver solves the resulting linear system to recover it without asking for a retransmission. The window adapts to the receiver's acknowledgments, so unlike block codes there is no fixed batch the receiver has to wait out before recovery becomes possible.

## What it is good for

Links where retransmission is too expensive to wait out: real-time video and audio, satellite, very long round-trips, multicast, IoT over noisy radio. delp sits between an application and UDP, recovers most of the channel loss in software, and stops adding redundancy when the channel is clean.

## Status

The library compiles, runs, and passes its full test suite (codec round trips, wire format, SIMD kernels, property tests, stress tests, async transport, long-session Cauchy). It cross-compiles for aarch64. CI runs format, clippy with warnings denied, every feature combination, an MSRV build, and a documentation build.

What is not in the box yet: a fuzz harness against the wire-format parsers, a `cargo-audit` integration in CI, multi-version MSRV testing, and a deployment-grade audit by anyone other than the author. The public API is not yet stable; expect breaking changes before a 1.0 tag.

## Install

Library:

    [dependencies]
    delp = "1"

With the optional async UDP transport (`FecSender`, `FecReceiver`, `DelpSession`):

    delp = { version = "1", features = ["async"] }

CLI:

    cargo install --path delp-cli

## CLI

The `delp` binary measures the codec on your hardware and exercises the protocol over a real socket without any Rust glue.

**Throughput.** Encodes and decodes a synthetic stream end-to-end, prints MB/s for each side.

    delp bench --symbol-size 1024 --window 64 --symbols 10000
    delp bench --symbol-size 1024 --window 32 --symbols 10000 --strategy cauchy

**File transfer over UDP.** Two terminals.

Receiver:

    delp recv --bind 0.0.0.0:9000 --output received.bin

Sender:

    delp send --file input.bin --dest 127.0.0.1:9000 --loss-rate 0.10

The sender drops a configurable fraction of source packets before they hit the wire and announces the file's SHA-256 in a session-start frame. The receiver verifies the digest before writing the output, so a successful run is a byte-exact recovery, not a "looks fine" recovery.

Most useful flags:

| Flag | Effect |
|---|---|
| `--symbol-size N` | Bytes per FEC symbol (default 1024). |
| `--window N` | Sliding-window capacity (default 32). |
| `--fec N:D` | Redundancy ratio, `1:1` = one coded packet per source packet. |
| `--strategy {vandermonde,cauchy}` | Coefficient strategy. |
| `--altc {none,recent,per-receiver}` | Adaptive subset coding (see below). |
| `--altc-recent N` | Under `--altc recent`, cover the last N symbols. |
| `--loss-rate F` | Drop a fraction of source packets at the sender. |

**Demos.** Two subcommands prove the two delp-specific extensions on stdout, no network needed.

    delp demo altc

Generates a full-window coded packet and an ALTC-targeted coded packet from the same encoder state, then prints the wire sizes, the source-ID counts in the encoding vector, and the resulting decoder row weight. With Cauchy the wire size shrinks linearly with the cover; with Vandermonde the matrix-row weight drops by the same ratio.

    delp demo generation --coded 1200

Runs a Cauchy session through 1 200 coded packets — well past the per-cycle limit — and prints the generation counter rolling over while the decoder keeps accepting packets without an error.

## Features

Library:

- GF(2⁴) and GF(2⁸) finite-field arithmetic with compile-time log/exp tables (no `OnceLock` initialisation cost).
- SIMD multiply-accumulate kernels for the hot path: AVX2 (64 B/iter, 2× unrolled), SSSE3 (16 B/iter), aarch64 NEON (16 B/iter via `vqtbl1q_u8`), scalar fallback. Runtime dispatch.
- Two coefficient strategies switchable per encoder: Vandermonde and Cauchy.
- Pluggable strategy traits for window eviction, congestion control, FEC rate, and feedback scheduling.
- Multi-receiver ACK tracking with cumulative + selective acknowledgment encoding.
- Optional async UDP transport built on `tokio::net::UdpSocket`, with `Stream` and `Sink` impls.
- Backpressure flow control — the encoder can return `WindowFull` to the caller instead of evicting unacknowledged symbols.

Two extensions on top of the standard sliding-window construction:

- **Generation rotation.** Cauchy GF(2⁸) admits 128 distinct coded IDs per cycle because the y-point set has 128 disjoint values. delp adds a one-byte generation field to the encoding vector. The y-point assignment rotates by `(coded_id + gen·K) mod cycle` with `K` coprime to the cycle, so each generation produces a fresh batch of 128 linearly-independent coded packets. With a u8 counter that is at least 32 768 coded packets per session before the encoder needs a session reset. The wire change is backward compatible: when generation is zero the encoding vector is byte-identical to the base format.
- **Adaptive Loss-Targeted Coding.** A coded packet does not have to cover the entire window. The encoder exposes three subset selectors: an explicit list of source IDs, the most-recent N symbols, or the IDs that a specific receiver has not yet acknowledged. Smaller covers translate directly into fewer GF multiplications at the encoder, sparser rows at the decoder, and (under Cauchy, where the coefficient list is sent explicitly) shorter packets on the wire.

## Performance

Numbers below are from `delp bench` on a single x86-64 machine with AVX2 detected at runtime, 1024-byte symbols, in-process loopback (no UDP, no loss). Treat as orientation, not as benchmark results.

| Strategy    | Encode    | Decode   |
|-------------|-----------|----------|
| Vandermonde | ~290 MB/s | ~35 MB/s |
| Cauchy      | ~480 MB/s | ~20 MB/s |

The encode path is dominated by the SIMD multiply-accumulate; the numbers track vector width on different hardware. Decode is bound by the Gaussian-elimination row substitution and is the main optimisation target left in the codec.

## Tests

`cargo test --workspace --all-features` runs 136 tests grouped as:

- Codec, wire format, and GF axioms (~100 tests inside the library).
- Async UDP transport, end-to-end on loopback.
- Property tests over random loss patterns and SIMD kernels (12 tests, proptest).
- Stress tests with random, bursty, and head-of-stream loss patterns.
- A 1 200-packet Cauchy session that crosses the per-cycle boundary.
- ALTC subset-coding integration tests.

## License

Apache-2.0.
