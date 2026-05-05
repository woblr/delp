use std::collections::VecDeque;
use bytes::Bytes;
use crate::policy::SourceSymbolId;

/// One entry in the encoding window.
#[derive(Debug, Clone)]
pub struct WindowSymbol {
    pub id:   SourceSymbolId,
    pub data: Bytes,
}

/// Elastic sliding window of source symbols held by the encoder.
///
/// Internally a `VecDeque` for O(1) front/back operations.
/// The window tracks whether it is *contiguous* so the encoder can select
/// the bandwidth-optimal `IdStorageFormat::None` wire encoding.
#[derive(Debug)]
pub struct EncodingWindow {
    symbols:    VecDeque<WindowSymbol>,
    capacity:   usize,
    /// ID that will be assigned to the next submitted symbol.
    pub(crate) next_id: SourceSymbolId,
    /// True when all IDs in the window form a contiguous range.
    pub(crate) is_contiguous: bool,
}

impl EncodingWindow {
    pub fn new(capacity: usize) -> Self {
        Self {
            symbols:       VecDeque::with_capacity(capacity.min(256)),
            capacity,
            next_id:       0,
            is_contiguous: true,
        }
    }

    // ── Accessors ────────────────────────────────────────────────────────

    pub fn len(&self)      -> usize { self.symbols.len() }
    pub fn is_empty(&self) -> bool  { self.symbols.is_empty() }
    pub fn is_full(&self)  -> bool  { self.symbols.len() >= self.capacity }

    pub fn first_id(&self) -> Option<SourceSymbolId> {
        self.symbols.front().map(|s| s.id)
    }

    pub fn last_id(&self) -> Option<SourceSymbolId> {
        self.symbols.back().map(|s| s.id)
    }

    /// Return a slice of all (id, data) pairs — used for coded-packet generation.
    pub fn symbols(&self) -> &VecDeque<WindowSymbol> { &self.symbols }

    /// Return a sorted `Vec` of all current source IDs — used by policy callbacks.
    pub fn id_slice(&self) -> Vec<SourceSymbolId> {
        self.symbols.iter().map(|s| s.id).collect()
    }

    // ── Mutation ─────────────────────────────────────────────────────────

    /// Push a new symbol and return its assigned ID.
    ///
    /// Panics if the window is full (caller must check `is_full()` first).
    pub fn push(&mut self, data: Bytes) -> SourceSymbolId {
        debug_assert!(!self.is_full());
        let id = self.next_id;
        // Check if we would break contiguity (only possible on wrapping, unlikely
        // in practice but we track it for correctness).
        if let Some(last) = self.last_id() {
            if id != last.wrapping_add(1) {
                self.is_contiguous = false;
            }
        }
        self.symbols.push_back(WindowSymbol { id, data });
        self.next_id = self.next_id.wrapping_add(1);
        id
    }

    /// Evict the oldest symbol from the front of the window.
    pub fn evict_oldest(&mut self) {
        self.symbols.pop_front();
        // After eviction the window is contiguous again iff it was before
        // (we only ever push to the back, evict from the front).
    }

    /// Remove specific source IDs from the window (mid-window removal for SACK).
    ///
    /// After removal the window may no longer be contiguous.
    pub fn remove_ids(&mut self, ids: &[SourceSymbolId]) {
        if ids.is_empty() { return; }

        let id_set: std::collections::HashSet<SourceSymbolId> =
            ids.iter().copied().collect();

        let before_len = self.symbols.len();
        self.symbols.retain(|s| !id_set.contains(&s.id));
        let removed = before_len - self.symbols.len();

        if removed > 0 {
            // After mid-window removal check contiguity
            self.is_contiguous = check_contiguous(&self.symbols);
        }
    }

    /// Return the payload for a specific source ID, if it is still in the window.
    ///
    /// O(1) if the window is contiguous (index by offset from front ID);
    /// O(n) scan for non-contiguous windows (rare).
    pub fn get(&self, id: SourceSymbolId) -> Option<&Bytes> {
        if self.is_empty() { return None; }

        if self.is_contiguous {
            let front = self.symbols.front().unwrap().id;
            if id < front { return None; }
            let idx = (id - front) as usize;
            self.symbols.get(idx).map(|s| &s.data)
        } else {
            self.symbols.iter().find(|s| s.id == id).map(|s| &s.data)
        }
    }
}

fn check_contiguous(symbols: &VecDeque<WindowSymbol>) -> bool {
    symbols.iter().zip(symbols.iter().skip(1))
        .all(|(a, b)| b.id == a.id.wrapping_add(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bytes(v: u8) -> Bytes { Bytes::from(vec![v; 8]) }

    #[test]
    fn push_and_get() {
        let mut w = EncodingWindow::new(8);
        let id0 = w.push(make_bytes(0));
        let id1 = w.push(make_bytes(1));
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(w.get(0).unwrap()[0], 0);
        assert_eq!(w.get(1).unwrap()[0], 1);
    }

    #[test]
    fn contiguous_flag_cleared_on_mid_removal() {
        let mut w = EncodingWindow::new(8);
        for i in 0..6u8 { w.push(make_bytes(i)); }
        assert!(w.is_contiguous);
        w.remove_ids(&[2, 4]);
        assert!(!w.is_contiguous);
    }

    #[test]
    fn evict_oldest_maintains_contiguity() {
        let mut w = EncodingWindow::new(4);
        for i in 0..4u8 { w.push(make_bytes(i)); }
        w.evict_oldest();
        assert!(w.is_contiguous);
        assert_eq!(w.first_id(), Some(1));
    }

    #[test]
    fn capacity_enforcement() {
        let mut w = EncodingWindow::new(3);
        for i in 0..3u8 { w.push(make_bytes(i)); }
        assert!(w.is_full());
    }
}