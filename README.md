# delp

Forward error correction library for Rust. Sends repair packets so the receiver can recover lost data without asking for retransmission.

The encoding window adjusts based on feedback from the receiver, so it works well on links where loss is variable.

## Install

```toml
[dependencies]
delp = "1.0"
```

## License

Apache-2.0