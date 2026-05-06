//! `delp recv` — receives a file sent by `delp send`.
//!
//! Listens for the START control frame, builds a delp decoder matched to
//! that session, then absorbs source + coded packets until the END frame
//! arrives or the inactivity timeout fires.  On EOF the SHA-256 of the
//! reconstructed file is compared against the digest from the START
//! frame; mismatch is reported as an error.

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context};
use bytes::Bytes;
use sha2::{Digest, Sha256};
use socket2::{Domain, Protocol, Socket, Type};
use tokio::net::UdpSocket;
use tokio::time::timeout;

use delp::codec::decoder::Decoder;
use delp::codec::DecoderEvent;
use delp::config::DecoderConfig;
use delp::policy::defaults::ConstantFeedbackPolicy;
use delp::wire::{coded::CodedPacket, source::SourcePacket};

use crate::send::{CTRL_END, CTRL_START};

const MAGIC: &[u8; 4] = b"DELP";
const RX_BUF: usize = 65_536;

struct Session {
    symbol_size: usize,
    n_symbols: u32,
    file_size: u64,
    sha256: [u8; 32],
}

fn parse_start(buf: &[u8]) -> anyhow::Result<Session> {
    if buf.len() < 4 + 4 + 4 + 8 + 32 {
        anyhow::bail!("START frame too short ({} B)", buf.len());
    }
    if buf[0..2] != MAGIC[0..2] {
        anyhow::bail!("missing DELP magic");
    }
    // byte 2 is symbol_size high byte but byte 0..2 already matched
    // the first two bytes of MAGIC ("DE") — re-validate the full
    // expected layout.
    // Layout (see send.rs):
    //   [0..4]  'D' 'E' symbol_size_hi  CTRL_START (=0xFF)
    //   wait, that's not right.  Let me follow the sender precisely.
    // Sender writes MAGIC (4B), then version(1) reserved(1) symsize(2),
    // then overwrites buf[3] = CTRL_START (so byte 3 carries the type).
    // So the wire layout is:
    //   [0]='D' [1]='E' [2]='L' [3]=0xFF (CTRL_START)
    //   [4]=ver [5]=0   [6..8]=symbol_size BE
    //   [8..12]=n_symbols BE
    //   [12..20]=file_size BE
    //   [20..52]=sha256
    if buf[0] != b'D' || buf[1] != b'E' || buf[2] != b'L' || buf[3] != CTRL_START {
        anyhow::bail!("malformed START frame header");
    }
    let version = buf[4];
    if version != 1 {
        anyhow::bail!("unsupported START version {version}");
    }
    let symbol_size = u16::from_be_bytes([buf[6], buf[7]]) as usize;
    let n_symbols = u32::from_be_bytes(buf[8..12].try_into().unwrap());
    let file_size = u64::from_be_bytes(buf[12..20].try_into().unwrap());
    let mut sha256 = [0u8; 32];
    sha256.copy_from_slice(&buf[20..52]);
    Ok(Session {
        symbol_size,
        n_symbols,
        file_size,
        sha256,
    })
}

pub async fn run(bind: SocketAddr, output: PathBuf, inactivity: Duration) -> anyhow::Result<()> {
    // Build the socket via socket2 so we can crank up SO_RCVBUF before
    // the kernel hands us the bound fd.  Loopback floods at >100 MB/s,
    // and the default 208 KB receive buffer drops packets long before
    // the user-space loop can drain them.
    let domain = match bind {
        SocketAddr::V4(_) => Domain::IPV4,
        SocketAddr::V6(_) => Domain::IPV6,
    };
    let s2 = Socket::new(domain, Type::DGRAM, Some(Protocol::UDP))
        .with_context(|| format!("creating UDP socket for {bind}"))?;
    // 16 MB receive buffer — kernel may cap to net.core.rmem_max.
    s2.set_recv_buffer_size(16 * 1024 * 1024).ok();
    s2.set_nonblocking(true)?;
    s2.bind(&bind.into())
        .with_context(|| format!("binding {bind}"))?;
    let std_sock: std::net::UdpSocket = s2.into();
    let socket = UdpSocket::from_std(std_sock)?;
    let actual = socket.local_addr()?;
    println!("delp recv  listening on {actual} → {}", output.display());

    let mut buf = vec![0u8; RX_BUF];

    // 1) Wait for the START frame.
    let session = loop {
        let (n, src) = timeout(inactivity, socket.recv_from(&mut buf))
            .await
            .map_err(|_| anyhow!("timed out waiting for START"))??;
        if n >= 4 && buf[3] == CTRL_START {
            let s = parse_start(&buf[..n])?;
            println!(
                "  session: {} symbols × {} B = {} bytes  (peer {src})",
                s.n_symbols, s.symbol_size, s.file_size
            );
            break s;
        }
    };

    // 2) Build the matched decoder.
    let dec_cfg = DecoderConfig::builder(session.symbol_size).build()?;
    let mut decoder = Decoder::new(dec_cfg, ConstantFeedbackPolicy::new(8));

    let mut symbols: BTreeMap<u32, Bytes> = BTreeMap::new();
    let mut got_end = false;
    let mut last_progress: Instant;
    let mut pkts_in = 0u64;

    // 3) Receive loop.
    while !got_end {
        let recv = timeout(inactivity, socket.recv_from(&mut buf)).await;
        let (n, _src) = match recv {
            Ok(r) => r?,
            Err(_) => {
                // Inactivity timeout: if we already have all symbols, fall through.
                if symbols.len() as u32 >= session.n_symbols {
                    break;
                }
                anyhow::bail!(
                    "inactivity timeout after {:.1}s — got {}/{} symbols",
                    inactivity.as_secs_f64(),
                    symbols.len(),
                    session.n_symbols
                );
            }
        };
        if n < 4 {
            continue;
        }
        last_progress = Instant::now();
        pkts_in += 1;

        match buf[3] {
            CTRL_END => {
                got_end = true;
            }
            0x00 => {
                // Source packet.
                if let Ok(sp) = SourcePacket::parse(&buf[..n]) {
                    if let Ok(events) = decoder.handle_source(&sp) {
                        for ev in events {
                            if let DecoderEvent::SourceReady { id, data } = ev {
                                symbols.insert(id, data);
                            }
                        }
                    }
                }
            }
            0x01 => {
                // Coded packet.
                if let Ok(cp) = CodedPacket::parse(&buf[..n]) {
                    if let Ok(events) = decoder.handle_coded(&cp) {
                        for ev in events {
                            if let DecoderEvent::SourceReady { id, data } = ev {
                                symbols.insert(id, data);
                            }
                        }
                    }
                }
            }
            0x02 => { /* feedback aimed at us is meaningless on the rx side */ }
            _ => { /* unknown / control frames not for this side */ }
        }

        if symbols.len() as u32 >= session.n_symbols {
            // Wait briefly for END (or until inactivity).
            if last_progress.elapsed() > Duration::from_millis(150) {
                break;
            }
        }
    }

    // 4) Stitch the file together.
    let mut out_buf = Vec::with_capacity(session.file_size as usize);
    for id in 0..session.n_symbols {
        match symbols.get(&id) {
            Some(data) => out_buf.extend_from_slice(data),
            None => anyhow::bail!("missing symbol {id} after END"),
        }
    }
    out_buf.truncate(session.file_size as usize);

    // 5) Verify SHA-256.
    let got = {
        let mut h = Sha256::new();
        h.update(&out_buf);
        let d = h.finalize();
        let mut a = [0u8; 32];
        a.copy_from_slice(&d);
        a
    };

    std::fs::write(&output, &out_buf).with_context(|| format!("writing {}", output.display()))?;

    if got == session.sha256 {
        println!(
            "✓ {} bytes received and verified ({} packets in)",
            out_buf.len(),
            pkts_in
        );
        Ok(())
    } else {
        Err(anyhow!(
            "SHA-256 mismatch — wrote {} but checksum does not match",
            output.display()
        ))
    }
}
