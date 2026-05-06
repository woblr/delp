//! `delp` — command-line tool for the delp FEC library.
//!
//! Transfer a file over a lossy UDP link, measure codec throughput,
//! and run live demos of delp's ALTC and Generation-Rotation extensions.

use clap::{Parser, Subcommand, ValueEnum};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

mod bench;
mod demo;
mod recv;
mod send;

#[derive(Parser)]
#[command(name = "delp", version, about = "delp FEC — CLI test harness")]
#[command(long_about = "\
delp is a pure-Rust forward-error-correction library.  This CLI exposes:

  send / recv      file transfer over UDP with simulated packet loss
  bench            encoder + decoder throughput (MB/s)
  demo altc        live wire-size proof of Adaptive Loss-Targeted Coding
  demo generation  multi-cycle Cauchy session past the RFC 9407 cap
")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
pub enum Strategy {
    Vandermonde,
    Cauchy,
}

impl Strategy {
    pub fn into_delp(self) -> delp::config::MatrixStrategy {
        match self {
            Strategy::Vandermonde => delp::config::MatrixStrategy::Vandermonde,
            Strategy::Cauchy => delp::config::MatrixStrategy::Cauchy,
        }
    }
}

#[derive(ValueEnum, Clone, Copy, Debug)]
pub enum AltcMode {
    /// Cover the entire window (RFC 9407 default).
    None,
    /// Cover the most-recent N symbols (`--altc-recent N`).
    Recent,
    /// Per-receiver coding: only symbols not yet ACK'd by the receiver.
    PerReceiver,
}

#[derive(Subcommand)]
enum Command {
    /// Send a file through delp over UDP, optionally simulating loss.
    Send {
        #[arg(long)]
        file: PathBuf,
        #[arg(long)]
        dest: SocketAddr,
        #[arg(long, default_value = "1024")]
        symbol_size: usize,
        #[arg(long, default_value = "32")]
        window: usize,
        /// FEC redundancy ratio numer:denom — e.g. "1:2" = one coded per two source.
        #[arg(long, default_value = "1:2")]
        fec: String,
        /// Galois field strategy.
        #[arg(long, value_enum, default_value_t = Strategy::Vandermonde)]
        strategy: Strategy,
        /// Adaptive Loss-Targeted Coding mode.
        #[arg(long, value_enum, default_value_t = AltcMode::None)]
        altc: AltcMode,
        /// `--altc-recent N` is only used when `--altc recent`.
        #[arg(long, default_value = "8")]
        altc_recent: usize,
        /// Drop a fraction of source packets before they hit the wire (0.0..=1.0).
        #[arg(long, default_value = "0.0")]
        loss_rate: f64,
    },
    /// Receive a file sent by `delp send`.
    Recv {
        #[arg(long)]
        bind: SocketAddr,
        #[arg(long)]
        output: PathBuf,
        /// Inactivity timeout in seconds before bailing out.
        #[arg(long, default_value = "30")]
        timeout_secs: u64,
    },
    /// Encoder + decoder throughput benchmark.
    Bench {
        #[arg(long, default_value = "1024")]
        symbol_size: usize,
        #[arg(long, default_value = "64")]
        window: usize,
        #[arg(long, default_value = "10000")]
        symbols: usize,
        #[arg(long, value_enum, default_value_t = Strategy::Vandermonde)]
        strategy: Strategy,
    },
    /// Live demos of delp-specific features.
    Demo {
        #[command(subcommand)]
        what: DemoCmd,
    },
}

#[derive(Subcommand)]
enum DemoCmd {
    /// Compare wire size and matrix-row weight: full-window vs ALTC.
    Altc {
        #[arg(long, default_value = "32")]
        window: usize,
        #[arg(long, default_value = "1024")]
        symbol_size: usize,
        #[arg(long, default_value = "8")]
        cover: usize,
    },
    /// Drive a Cauchy session past the RFC 9407 128-packet cap and prove
    /// that delp's generation-rotation extension keeps it producing
    /// linearly-independent coded packets.
    Generation {
        #[arg(long, default_value = "12")]
        symbols: usize,
        #[arg(long, default_value = "1200")]
        coded: u32,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Command::Send {
            file,
            dest,
            symbol_size,
            window,
            fec,
            strategy,
            altc,
            altc_recent,
            loss_rate,
        } => {
            let (fec_n, fec_d) = parse_fec_ratio(&fec)?;
            send::run(send::Config {
                file,
                dest,
                symbol_size,
                window,
                fec_n,
                fec_d,
                strategy,
                altc,
                altc_recent,
                loss_rate,
            })
            .await
        }
        Command::Recv {
            bind,
            output,
            timeout_secs,
        } => recv::run(bind, output, Duration::from_secs(timeout_secs)).await,
        Command::Bench {
            symbol_size,
            window,
            symbols,
            strategy,
        } => bench::run(symbol_size, window, symbols, strategy),
        Command::Demo { what } => match what {
            DemoCmd::Altc {
                window,
                symbol_size,
                cover,
            } => demo::altc(window, symbol_size, cover),
            DemoCmd::Generation { symbols, coded } => demo::generation(symbols, coded),
        },
    }
}

fn parse_fec_ratio(s: &str) -> anyhow::Result<(usize, usize)> {
    let mut parts = s.splitn(2, ':');
    let n: usize = parts
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing FEC numerator"))?
        .trim()
        .parse()?;
    let d: usize = parts
        .next()
        .ok_or_else(|| anyhow::anyhow!("FEC ratio must be N:D, e.g. 1:2"))?
        .trim()
        .parse()?;
    if d == 0 {
        anyhow::bail!("FEC denominator must be > 0");
    }
    Ok((n, d))
}
