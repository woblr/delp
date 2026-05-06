use crate::policy::SourceSymbolId;
use bytes::Bytes;
use std::collections::BTreeMap;

/// In-order symbol delivery buffer for the decoder.
///
/// Stores received and recovered source symbols, then drains them in
/// ascending order starting from `next_delivery_id`.
#[derive(Debug, Default)]
pub struct SymbolBuffer {
    /// Received or recovered source symbols, keyed by ID.
    symbols: BTreeMap<SourceSymbolId, Bytes>,
    /// Next ID the application expects to receive in order.
    pub(crate) next_delivery_id: SourceSymbolId,
}

impl SymbolBuffer {
    pub fn new(start_id: SourceSymbolId) -> Self {
        Self {
            symbols: BTreeMap::new(),
            next_delivery_id: start_id,
        }
    }

    // ── Write ────────────────────────────────────────────────────────────

    /// Store a symbol.  Silently drops duplicates (already-delivered IDs).
    pub fn insert(&mut self, id: SourceSymbolId, data: Bytes) {
        if id >= self.next_delivery_id {
            self.symbols.entry(id).or_insert(data);
        }
    }

    // ── Read ─────────────────────────────────────────────────────────────

    pub fn contains(&self, id: SourceSymbolId) -> bool {
        self.symbols.contains_key(&id)
    }

    pub fn get(&self, id: SourceSymbolId) -> Option<&Bytes> {
        self.symbols.get(&id)
    }

    /// The highest source ID currently buffered, if any.
    pub fn highest_id(&self) -> Option<SourceSymbolId> {
        self.symbols.keys().next_back().copied()
    }

    // ── Delivery ─────────────────────────────────────────────────────────

    /// Drain and return all contiguous symbols starting from `next_delivery_id`.
    ///
    /// Advances `next_delivery_id` past every delivered symbol.
    pub fn drain_deliverable(&mut self) -> Vec<(SourceSymbolId, Bytes)> {
        let mut out = Vec::new();
        while let Some(data) = self.symbols.remove(&self.next_delivery_id) {
            out.push((self.next_delivery_id, data));
            self.next_delivery_id = self.next_delivery_id.wrapping_add(1);
        }
        out
    }

    // ── Pruning ──────────────────────────────────────────────────────────

    /// Remove all symbols with IDs strictly below `floor`.
    ///
    /// Called when the encoder signals that its window has advanced past
    /// a point where these symbols can no longer be recovered anyway.
    pub fn prune_below(&mut self, floor: SourceSymbolId) {
        // BTreeMap::retain is O(n) but we expect infrequent pruning.
        self.symbols.retain(|&id, _| id >= floor);
        if self.next_delivery_id < floor {
            self.next_delivery_id = floor;
        }
    }

    /// IDs of buffered symbols starting from `from` (for feedback generation).
    pub fn ids_from(&self, from: SourceSymbolId) -> impl Iterator<Item = SourceSymbolId> + '_ {
        self.symbols.range(from..).map(|(&id, _)| id)
    }

    pub fn len(&self) -> usize {
        self.symbols.len()
    }
    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(v: u8) -> Bytes {
        Bytes::from(vec![v; 4])
    }

    #[test]
    fn in_order_delivery() {
        let mut buf = SymbolBuffer::new(0);
        buf.insert(0, bytes(0));
        buf.insert(1, bytes(1));
        buf.insert(2, bytes(2));
        let delivered = buf.drain_deliverable();
        assert_eq!(delivered.len(), 3);
        assert_eq!(delivered[0].0, 0);
        assert_eq!(delivered[2].0, 2);
        assert_eq!(buf.next_delivery_id, 3);
    }

    #[test]
    fn out_of_order_held_until_gap_filled() {
        let mut buf = SymbolBuffer::new(0);
        buf.insert(0, bytes(0));
        buf.insert(2, bytes(2)); // gap at 1
        let d1 = buf.drain_deliverable();
        assert_eq!(d1.len(), 1); // only 0 delivered
        buf.insert(1, bytes(1)); // fill gap
        let d2 = buf.drain_deliverable();
        assert_eq!(d2.len(), 2); // 1 and 2 now delivered
    }

    #[test]
    fn prune_below_advances_delivery_ptr() {
        let mut buf = SymbolBuffer::new(0);
        buf.insert(5, bytes(5));
        buf.prune_below(5);
        assert_eq!(buf.next_delivery_id, 5);
        let d = buf.drain_deliverable();
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].0, 5);
    }

    #[test]
    fn duplicate_insert_ignored() {
        let mut buf = SymbolBuffer::new(0);
        buf.insert(0, bytes(0xAA));
        buf.insert(0, bytes(0xBB)); // duplicate
        let d = buf.drain_deliverable();
        assert_eq!(d[0].1[0], 0xAA); // first insert wins
    }
}
