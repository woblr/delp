use bytes::Bytes;
use std::collections::BTreeMap;

use crate::config::Field;
use crate::error::{Result, DelpError};
use crate::gf::simd::ops::{
    mul_acc_gf2_8, mul_acc_gf2_4,
    mul_scale_gf2_8, mul_scale_gf2_4,
};
use crate::policy::SourceSymbolId;
use crate::wire::ev::EncodingVector;

// ── Dense coefficient row ─────────────────────────────────────────────────

/// A row in the decoding matrix stored as a dense `Vec<u8>` indexed by
/// `(source_id - base_id)`.  Dense storage gives O(1) access and makes
/// `subtract_scaled` a tight SIMD loop instead of a BTreeMap walk.
#[derive(Debug, Clone)]
struct DenseCoefs {
    /// Smallest source ID covered by this row.
    base_id: SourceSymbolId,
    /// `coefs[i]` = coefficient for source `base_id + i`.  Zero entries
    /// are explicit (sparse compression would break the SIMD pattern).
    coefs: Vec<u8>,
}

impl DenseCoefs {
    fn new(base_id: SourceSymbolId, len: usize) -> Self {
        Self { base_id, coefs: vec![0u8; len] }
    }

    fn from_map(map: &BTreeMap<SourceSymbolId, u8>) -> Self {
        if map.is_empty() {
            return Self { base_id: 0, coefs: Vec::new() };
        }
        let &min = map.keys().next().unwrap();
        let &max = map.keys().next_back().unwrap();
        let len  = (max - min) as usize + 1;
        let mut c = Self::new(min, len);
        for (&id, &coef) in map {
            c.coefs[(id - min) as usize] = coef;
        }
        c
    }

    #[inline]
    fn get(&self, id: SourceSymbolId) -> u8 {
        if id < self.base_id { return 0; }
        let idx = (id - self.base_id) as usize;
        *self.coefs.get(idx).unwrap_or(&0)
    }

    #[inline]
    #[allow(dead_code)]
    fn set(&mut self, id: SourceSymbolId, val: u8) {
        self.ensure_covers(id);
        self.coefs[(id - self.base_id) as usize] = val;
    }

    fn ensure_covers(&mut self, id: SourceSymbolId) {
        if self.coefs.is_empty() {
            self.base_id = id;
            self.coefs.push(0);
            return;
        }
        if id < self.base_id {
            let prepend = (self.base_id - id) as usize;
            let mut new = vec![0u8; prepend + self.coefs.len()];
            new[prepend..].copy_from_slice(&self.coefs);
            self.coefs  = new;
            self.base_id = id;
        } else {
            let needed = (id - self.base_id) as usize + 1;
            if needed > self.coefs.len() {
                self.coefs.resize(needed, 0);
            }
        }
    }

    /// XOR-accumulate `factor * other` into self (in-place), using SIMD.
    fn subtract_scaled(&mut self, other: &DenseCoefs, factor: u8, field: Field) {
        if other.coefs.is_empty() || factor == 0 { return; }
        let other_end = other.base_id + other.coefs.len() as u32;
        self.ensure_covers(other.base_id);
        self.ensure_covers(other_end.saturating_sub(1));

        let dst_off = (other.base_id - self.base_id) as usize;
        let len     = other.coefs.len();
        let dst     = &mut self.coefs[dst_off..dst_off + len];

        match field {
            Field::Gf2_8 => mul_acc_gf2_8(dst, &other.coefs, factor),
            Field::Gf2_4 => mul_acc_gf2_4(dst, &other.coefs, factor),
        }
    }

    /// Scale all coefficients in-place by `factor`.
    fn scale(&mut self, factor: u8, field: Field) {
        match field {
            Field::Gf2_8 => mul_scale_gf2_8(&mut self.coefs, factor),
            Field::Gf2_4 => mul_scale_gf2_4(&mut self.coefs, factor),
        }
    }

    /// True when at most one non-zero coefficient remains (= recovery).
    fn is_solved(&self) -> Option<SourceSymbolId> {
        let mut pivot = None;
        for (i, &c) in self.coefs.iter().enumerate() {
            if c != 0 {
                if pivot.is_some() { return None; }
                pivot = Some(self.base_id + i as u32);
            }
        }
        pivot
    }

    fn pivot_id(&self) -> Option<SourceSymbolId> {
        self.coefs.iter().enumerate()
            .find(|(_, &c)| c != 0)
            .map(|(i, _)| self.base_id + i as u32)
    }

    fn pivot_coef(&self) -> Option<u8> {
        self.coefs.iter().copied().find(|&c| c != 0)
    }

    fn remove(&mut self, id: SourceSymbolId) {
        if id < self.base_id { return; }
        let idx = (id - self.base_id) as usize;
        if idx < self.coefs.len() { self.coefs[idx] = 0; }
    }

    fn is_zero(&self) -> bool { self.coefs.iter().all(|&c| c == 0) }

    #[allow(dead_code)]
    fn nonzero_ids(&self) -> impl Iterator<Item = SourceSymbolId> + '_ {
        self.coefs.iter().enumerate()
            .filter(|(_, &c)| c != 0)
            .map(move |(i, _)| self.base_id + i as u32)
    }
}

// ── MatrixRow ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct MatrixRow {
    coefs:   DenseCoefs,
    payload: Vec<u8>,
}

impl MatrixRow {
    fn subtract_scaled(&mut self, other: &MatrixRow, factor: u8, field: Field) {
        self.coefs.subtract_scaled(&other.coefs, factor, field);
        // payload -= factor * other.payload  (same SIMD call)
        match field {
            Field::Gf2_8 => mul_acc_gf2_8(&mut self.payload, &other.payload, factor),
            Field::Gf2_4 => mul_acc_gf2_4(&mut self.payload, &other.payload, factor),
        }
    }

    fn normalize(&mut self, field: Field) -> Result<()> {
        let pivot_c = self.coefs.pivot_coef()
            .ok_or(DelpError::MalformedEncodingVector { reason: "zero pivot" })?;
        if pivot_c == 1 { return Ok(()); }

        // Compute inverse of pivot coefficient
        let inv = gf_inv(pivot_c, field)?;
        self.coefs.scale(inv, field);
        match field {
            Field::Gf2_8 => mul_scale_gf2_8(&mut self.payload, inv),
            Field::Gf2_4 => mul_scale_gf2_4(&mut self.payload, inv),
        }
        Ok(())
    }
}

// ── GF inverse helpers ────────────────────────────────────────────────────

fn gf_inv(v: u8, field: Field) -> Result<u8> {
    use crate::gf::{Field as GfField, Gf2_8, Gf2_4};
    if v == 0 { return Err(DelpError::DivisionByZero); }
    Ok(match field {
        Field::Gf2_8 => Gf2_8::new(v).inv().to_u8(),
        Field::Gf2_4 => Gf2_4::new(v & 0x0F).inv().to_u8(),
    })
}

#[allow(dead_code)]
fn gf_mul(a: u8, b: u8, field: Field) -> u8 {
    use crate::gf::{Field as GfField, Gf2_8, Gf2_4};
    match field {
        Field::Gf2_8 => Gf2_8::new(a).mul(Gf2_8::new(b)).to_u8(),
        Field::Gf2_4 => Gf2_4::new(a & 0x0F).mul(Gf2_4::new(b & 0x0F)).to_u8(),
    }
}

// ── DecodingMatrix ────────────────────────────────────────────────────────

/// Incremental Gaussian elimination decoding matrix.
///
/// Maintains a set of rows in reduced row-echelon form, keyed by pivot ID.
/// New equations are added one at a time; recovery is attempted after
/// each insertion (minimal latency — symbols are recovered as soon as
/// mathematically possible).
#[derive(Debug)]
pub struct DecodingMatrix {
    /// Pivot ID → row (row-echelon form invariant: each row has a unique pivot).
    rows:        BTreeMap<SourceSymbolId, MatrixRow>,
    field:       Field,
    #[allow(dead_code)]
    symbol_size: usize,
    max_rows:    usize,
}

impl DecodingMatrix {
    pub fn new(field: Field, symbol_size: usize, max_rows: usize) -> Self {
        Self {
            rows: BTreeMap::new(),
            field,
            symbol_size,
            max_rows,
        }
    }

    // ── Public interface ─────────────────────────────────────────────────

    /// Add a coded equation to the matrix.
    ///
    /// Returns a list of (source_id, payload) pairs for every newly recovered
    /// source symbol (may be empty, or cascade to multiple recoveries).
    pub fn add_coded(
        &mut self,
        ev:              &EncodingVector,
        payload:         &[u8],
        known_sources:   &std::collections::BTreeMap<SourceSymbolId, Bytes>,
    ) -> Result<Vec<(SourceSymbolId, Bytes)>> {
        // Build coefficient map from encoding vector
        let mut coef_map: BTreeMap<SourceSymbolId, u8> = BTreeMap::new();
        for (i, &id) in ev.source_ids.iter().enumerate() {
            let c = if ev.has_explicit_coefs() {
                ev.coefficients[i]
            } else {
                use crate::wire::ev::coef_gen::vandermonde_coef;
                vandermonde_coef(ev.field, id, ev.coded_id)
            };
            if c != 0 { coef_map.insert(id, c); }
        }

        let mut row = MatrixRow {
            coefs:   DenseCoefs::from_map(&coef_map),
            payload: payload.to_vec(),
        };

        // Substitute known sources
        let _newly_known: Vec<(SourceSymbolId, Bytes)> = Vec::new();
        for (&id, data) in known_sources {
            let c = row.coefs.get(id);
            if c == 0 { continue; }
            row.coefs.remove(id);
            let field = self.field;
            match field {
                Field::Gf2_8 => mul_acc_gf2_8(&mut row.payload, data, c),
                Field::Gf2_4 => mul_acc_gf2_4(&mut row.payload, data, c),
            }
        }

        if row.coefs.is_zero() { return Ok(Vec::new()); } // redundant equation

        // Forward-eliminate against existing pivot rows
        self.forward_eliminate(&mut row)?;

        if row.coefs.is_zero() { return Ok(Vec::new()); } // eliminated away

        // Normalise (pivot coefficient → 1)
        row.normalize(self.field)?;

        // Back-substitute: reduce all existing rows against the new pivot
        let pivot_id = row.coefs.pivot_id()
            .ok_or(DelpError::MalformedEncodingVector { reason: "no pivot after elimination" })?;

        self.back_substitute(pivot_id, &row)?;

        // Enforce max_rows limit before inserting
        if self.rows.len() >= self.max_rows {
            if let Some(oldest) = self.rows.keys().next().copied() {
                self.rows.remove(&oldest);
            }
        }

        self.rows.insert(pivot_id, row);

        // Collect all newly-single-coefficient rows (the new row may have
        // created singletons elsewhere via back-substitution)
        let mut all_recovered: Vec<(SourceSymbolId, Bytes)> = Vec::new();
        let mut extra_known: BTreeMap<SourceSymbolId, Bytes> = known_sources.clone();

        loop {
            // Find any row that is now a singleton
            let solved: Vec<SourceSymbolId> = self.rows.iter()
                .filter_map(|(&pid, row)| row.coefs.is_solved().map(|_| pid))
                .collect();

            if solved.is_empty() { break; }

            for pid in solved {
                if let Some(row) = self.rows.remove(&pid) {
                    if let Some(solved_id) = row.coefs.is_solved() {
                        let sym = Bytes::from(row.payload.clone());
                        all_recovered.push((solved_id, sym.clone()));
                        extra_known.insert(solved_id, sym.clone());
                        // Substitute this new knowledge into remaining rows
                        let more = self.add_known_source(solved_id, sym, &extra_known)?;
                        for (rid, rdata) in more {
                            extra_known.insert(rid, rdata.clone());
                            all_recovered.push((rid, rdata));
                        }
                    }
                }
            }
        }

        Ok(all_recovered)
    }

    /// Notify the matrix that a source symbol has been received/recovered.
    ///
    /// Substitutes it into all rows and cascades any resulting recoveries.
    ///
    /// Cascade is fully **iterative** (work queue, no recursion) so deep
    /// chains of dependent recoveries cannot overflow the stack regardless
    /// of window size or burst-loss depth.
    pub fn add_known_source(
        &mut self,
        id:            SourceSymbolId,
        data:          Bytes,
        known_sources: &std::collections::BTreeMap<SourceSymbolId, Bytes>,
    ) -> Result<Vec<(SourceSymbolId, Bytes)>> {
        let mut all_recovered: Vec<(SourceSymbolId, Bytes)> = Vec::new();

        // Work queue: (symbol_id, symbol_data) pairs still to be substituted.
        let mut queue: std::collections::VecDeque<(SourceSymbolId, Bytes)> =
            std::collections::VecDeque::new();
        queue.push_back((id, data));

        // Accumulated known symbols (initial set + everything we recover here).
        let mut local_known: BTreeMap<SourceSymbolId, Bytes> = known_sources.clone();

        while let Some((sym_id, sym_data)) = queue.pop_front() {
            // Skip if we already processed this symbol in a previous iteration
            // (can happen if the caller's known_sources set is large).
            if local_known.contains_key(&sym_id) {
                // Still substitute into matrix rows so they stay consistent.
            }

            let mut newly_solved: Vec<SourceSymbolId> = Vec::new();

            // Substitute sym_id into every row that references it.
            for row in self.rows.values_mut() {
                let c = row.coefs.get(sym_id);
                if c == 0 { continue; }
                row.coefs.remove(sym_id);
                let field = self.field;
                match field {
                    Field::Gf2_8 => mul_acc_gf2_8(&mut row.payload, &sym_data, c),
                    Field::Gf2_4 => mul_acc_gf2_4(&mut row.payload, &sym_data, c),
                }
                if let Some(solved_id) = row.coefs.is_solved() {
                    newly_solved.push(solved_id);
                }
            }

            local_known.insert(sym_id, sym_data);

            // Extract newly-solved rows and enqueue them for the next round.
            for solved_id in newly_solved {
                if let Some(row) = self.rows.remove(&solved_id) {
                    let sym = Bytes::from(row.payload);
                    all_recovered.push((solved_id, sym.clone()));
                    // Only enqueue if not already known — avoids duplicate work.
                    if !local_known.contains_key(&solved_id) {
                        queue.push_back((solved_id, sym));
                    }
                }
            }
        }

        Ok(all_recovered)
    }

    // ── Internals ────────────────────────────────────────────────────────

    fn forward_eliminate(&self, row: &mut MatrixRow) -> Result<()> {
        // `row` is an independent `&mut MatrixRow` — not part of `self.rows`.
        // Borrowing `&self.rows[pid]` (immutable) while mutating `row`
        // (mutable, disjoint) is safe; no clone needed.
        for pid in self.rows.keys().copied().collect::<Vec<_>>() {
            let c = row.coefs.get(pid);
            if c == 0 { continue; }
            let existing = &self.rows[&pid];
            row.subtract_scaled(existing, c, self.field);
        }

        // Partial pivoting: if after elimination the natural leftmost non-zero
        // column is zero (degenerate packet) we just return — the caller will
        // detect `coefs.is_zero()` and discard the row.  When the pivot coef
        // is non-zero but not 1, normalization (in add_coded) scales it to 1.
        // No explicit swap is needed here because each row is keyed by its
        // pivot ID in a BTreeMap — the ordering is maintained automatically.
        Ok(())
    }

    fn back_substitute(&mut self, new_pivot: SourceSymbolId, new_row: &MatrixRow) -> Result<()> {
        // `new_row` is the row we are about to insert — it is NOT yet in
        // `self.rows`, so borrowing it alongside `self.rows.get_mut` is safe.
        // We only clone the pivot-ID list (a handful of u32s) to satisfy
        // the borrow checker while iterating and mutating the map.
        for pid in self.rows.keys().copied().collect::<Vec<_>>() {
            let c = self.rows[&pid].coefs.get(new_pivot);
            if c == 0 { continue; }
            self.rows.get_mut(&pid).unwrap().subtract_scaled(new_row, c, self.field);
        }
        Ok(())
    }

    pub fn row_count(&self) -> usize { self.rows.len() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Field;
    use crate::wire::ev::EncodingVector;
    use smallvec::SmallVec;

    fn make_ev(field: Field, source_ids: &[u32], coded_id: u32) -> EncodingVector {
        let ids: SmallVec<[u32; 64]> = source_ids.iter().copied().collect();
        EncodingVector::vandermonde(field, coded_id, ids)
    }

    fn compute_coded_payload(field: Field, sources: &[(u32, Vec<u8>)], coded_id: u32) -> Vec<u8> {
        let sym_size = sources[0].1.len();
        let mut payload = vec![0u8; sym_size];
        for (src_id, data) in sources {
            let c = crate::wire::ev::coef_gen::vandermonde_coef(field, *src_id, coded_id);
            match field {
                Field::Gf2_8 => mul_acc_gf2_8(&mut payload, data, c),
                Field::Gf2_4 => mul_acc_gf2_4(&mut payload, data, c),
            }
        }
        payload
    }

    #[test]
    fn single_erasure_recovery_gf2_8() {
        let field     = Field::Gf2_8;
        let sym_size  = 64;
        let src0      = vec![0xAAu8; sym_size];
        let src1      = vec![0xBBu8; sym_size];
        let src2      = vec![0xCCu8; sym_size];
        let sources   = vec![(0u32, src0.clone()), (1u32, src1.clone()), (2u32, src2.clone())];

        // Simulate: src1 is lost; src0, src2 received; coded_1 received
        let coded_payload = compute_coded_payload(field, &sources, 1);
        let ev = make_ev(field, &[0, 1, 2], 1);

        let mut known: BTreeMap<u32, Bytes> = BTreeMap::new();
        known.insert(0, Bytes::from(src0.clone()));
        known.insert(2, Bytes::from(src2.clone()));

        let mut matrix = DecodingMatrix::new(field, sym_size, 256);
        let recovered = matrix.add_coded(&ev, &coded_payload, &known).unwrap();

        assert_eq!(recovered.len(), 1);
        let (rid, rdata) = &recovered[0];
        assert_eq!(*rid, 1);
        assert_eq!(rdata.as_ref(), src1.as_slice());
    }

    #[test]
    fn two_erasure_recovery_requires_two_coded() {
        let field    = Field::Gf2_8;
        let sym_size = 32;
        let src0     = (0..32u8).collect::<Vec<_>>();
        let src1     = (0..32u8).map(|x| x ^ 0xFF).collect::<Vec<_>>();
        let src2     = vec![0x55u8; 32];
        let sources  = vec![(0u32, src0.clone()), (1u32, src1.clone()), (2u32, src2.clone())];

        let c1 = compute_coded_payload(field, &sources, 1);
        let c2 = compute_coded_payload(field, &sources, 2);

        let mut known: BTreeMap<u32, Bytes> = BTreeMap::new();
        known.insert(2, Bytes::from(src2.clone()));

        let mut matrix = DecodingMatrix::new(field, sym_size, 256);
        let ev1 = make_ev(field, &[0, 1, 2], 1);
        let ev2 = make_ev(field, &[0, 1, 2], 2);

        let r1 = matrix.add_coded(&ev1, &c1, &known).unwrap();
        assert!(r1.is_empty(), "need second coded packet first");

        let r2 = matrix.add_coded(&ev2, &c2, &known).unwrap();
        assert_eq!(r2.len(), 2, "both erasures should be recovered");

        let recovered_map: BTreeMap<u32, Vec<u8>> =
            r2.into_iter().map(|(id, d)| (id, d.to_vec())).collect();
        assert_eq!(recovered_map[&0], src0);
        assert_eq!(recovered_map[&1], src1);
    }

    /// Full encoder→decoder round-trip with random packet loss.
    ///
    /// Sends N source symbols + FEC coded packets through a simulated channel
    /// that drops packets at a given loss rate.  Verifies that all source
    /// symbols are recovered when the redundancy is sufficient.
    fn round_trip_with_loss(
        field:     Field,
        n_src:     usize,
        sym_size:  usize,
        fec_rate:  f64, // e.g. 0.5 = one coded per two source
        loss_rate: f64, // e.g. 0.25 = 25 % drop
        seed:      u64,
    ) {
        use crate::wire::ev::coef_gen::vandermonde_coef;

        // ── Generate source symbols ──────────────────────────────────────
        let mut rng_state = seed;
        let mut lcg = move || -> u8 {
            rng_state = rng_state.wrapping_mul(6364136223846793005)
                                  .wrapping_add(1442695040888963407);
            (rng_state >> 33) as u8
        };

        let sources: Vec<Vec<u8>> = (0..n_src)
            .map(|_| (0..sym_size).map(|_| lcg()).collect())
            .collect();

        // ── Compute coded payloads (Vandermonde) ─────────────────────────
        let n_coded = (n_src as f64 * fec_rate).ceil() as usize;
        let coded_payloads: Vec<(u32, Vec<u8>)> = (1..=n_coded as u32)
            .map(|coded_id| {
                let mut payload = vec![0u8; sym_size];
                for (src_id, data) in sources.iter().enumerate() {
                    let c = vandermonde_coef(field, src_id as u32, coded_id);
                    match field {
                        Field::Gf2_8 => mul_acc_gf2_8(&mut payload, data, c),
                        Field::Gf2_4 => mul_acc_gf2_4(&mut payload, data, c),
                    }
                }
                (coded_id, payload)
            })
            .collect();

        // ── Simulate channel: drop packets at loss_rate ───────────────────
        let mut drop_state = seed ^ 0xDEADBEEF;
        let mut should_drop = move || -> bool {
            drop_state = drop_state.wrapping_mul(2862933555777941757)
                                    .wrapping_add(3037000499);
            let r = (drop_state >> 33) as f64 / u32::MAX as f64;
            r < loss_rate
        };

        let mut known: BTreeMap<u32, Bytes> = BTreeMap::new();
        let mut matrix = DecodingMatrix::new(field, sym_size, n_src + n_coded);

        // Deliver source packets (some dropped)
        for (src_id, data) in sources.iter().enumerate() {
            if !should_drop() {
                known.insert(src_id as u32, Bytes::from(data.clone()));
            }
        }

        // Deliver coded packets (some dropped)
        for (coded_id, payload) in &coded_payloads {
            if should_drop() { continue; }
            let src_ids: smallvec::SmallVec<[u32; 64]> =
                (0..n_src as u32).collect();
            let ev = make_ev(field, &(0..n_src as u32).collect::<Vec<_>>(), *coded_id);
            let recovered = matrix.add_coded(&ev, payload, &known).unwrap();
            for (id, data) in recovered {
                known.insert(id, data);
            }
        }

        // ── Verify all source symbols are known ───────────────────────────
        for (src_id, orig) in sources.iter().enumerate() {
            let pct = (loss_rate * 100.0) as u32;
            let recovered = known.get(&(src_id as u32))
                .unwrap_or_else(|| panic!(
                    "source {src_id} not recovered (loss={pct}%, seed={seed})"
                ));
            assert_eq!(recovered.as_ref(), orig.as_slice(),
                "source {src_id} data mismatch");
        }
    }

    #[test]
    fn round_trip_no_loss_gf2_8() {
        round_trip_with_loss(Field::Gf2_8, 8, 64, 0.5, 0.0, 42);
    }

    /// Controlled loss: exactly 2 of 8 sources dropped, 4 coded packets all delivered.
    /// With 4 coded = 4 equations and 2 unknowns, recovery is guaranteed.
    #[test]
    fn round_trip_2_erasures_guaranteed_recovery() {
        let field    = Field::Gf2_8;
        let n_src    = 8usize;
        let sym_size = 48usize;
        use crate::wire::ev::coef_gen::vandermonde_coef;

        let sources: Vec<Vec<u8>> = (0..n_src)
            .map(|i| (0..sym_size).map(|j| ((i * 17 + j) as u8)).collect())
            .collect();

        // All 4 coded packets
        let coded: Vec<(u32, Vec<u8>)> = (1..=4u32).map(|cid| {
            let mut p = vec![0u8; sym_size];
            for (sid, data) in sources.iter().enumerate() {
                let c = vandermonde_coef(field, sid as u32, cid);
                mul_acc_gf2_8(&mut p, data, c);
            }
            (cid, p)
        }).collect();

        // Drop sources 3 and 5 exactly
        let mut known: BTreeMap<u32, Bytes> = BTreeMap::new();
        for i in 0..n_src {
            if i == 3 || i == 5 { continue; }
            known.insert(i as u32, Bytes::from(sources[i].clone()));
        }

        let mut matrix = DecodingMatrix::new(field, sym_size, 64);
        for (cid, payload) in &coded {
            let ev = make_ev(field, &(0..n_src as u32).collect::<Vec<_>>(), *cid);
            let rec = matrix.add_coded(&ev, payload, &known).unwrap();
            for (id, data) in rec { known.insert(id, data); }
        }

        assert_eq!(known.get(&3).map(|b| b.as_ref()), Some(sources[3].as_slice()),
            "source 3 not recovered");
        assert_eq!(known.get(&5).map(|b| b.as_ref()), Some(sources[5].as_slice()),
            "source 5 not recovered");
    }

    #[test]
    fn round_trip_burst_loss_gf2_8() {
        // Simulate all first 4 of 8 sources dropped — FEC must recover them
        let field     = Field::Gf2_8;
        let n_src     = 8usize;
        let sym_size  = 32usize;
        use crate::wire::ev::coef_gen::vandermonde_coef;

        let sources: Vec<Vec<u8>> = (0..n_src)
            .map(|i| vec![(i as u8).wrapping_mul(17); sym_size])
            .collect();

        // Coded payloads
        let coded: Vec<(u32, Vec<u8>)> = (1..=4u32).map(|cid| {
            let mut p = vec![0u8; sym_size];
            for (sid, data) in sources.iter().enumerate() {
                let c = vandermonde_coef(field, sid as u32, cid);
                mul_acc_gf2_8(&mut p, data, c);
            }
            (cid, p)
        }).collect();

        // Only last 4 sources received
        let mut known: BTreeMap<u32, Bytes> = BTreeMap::new();
        for i in 4..n_src {
            known.insert(i as u32, Bytes::from(sources[i].clone()));
        }

        let mut matrix = DecodingMatrix::new(field, sym_size, 64);
        for (cid, payload) in &coded {
            let ev = make_ev(field, &(0..n_src as u32).collect::<Vec<_>>(), *cid);
            let rec = matrix.add_coded(&ev, payload, &known).unwrap();
            for (id, data) in rec { known.insert(id, data); }
        }

        for (sid, orig) in sources.iter().enumerate() {
            let got = known.get(&(sid as u32))
                .unwrap_or_else(|| panic!("source {sid} not recovered"));
            assert_eq!(got.as_ref(), orig.as_slice());
        }
    }

    #[test]
    fn redundant_equation_silently_discarded() {
        let field    = Field::Gf2_8;
        let sym_size = 16;
        let src0     = vec![1u8; 16];
        let sources  = vec![(0u32, src0.clone())];

        let c1 = compute_coded_payload(field, &sources, 1);
        let ev = make_ev(field, &[0], 1);
        let mut known: BTreeMap<u32, Bytes> = BTreeMap::new();

        let mut matrix = DecodingMatrix::new(field, sym_size, 256);
        let r1 = matrix.add_coded(&ev, &c1, &known).unwrap();
        assert_eq!(r1.len(), 1);

        // Submit same equation again → redundant
        known.insert(0, r1[0].1.clone());
        let r2 = matrix.add_coded(&ev, &c1, &known).unwrap();
        assert!(r2.is_empty());
    }
}