use crate::iter::TableIter;
use crate::types::Record;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Clone, Eq, PartialEq)]
struct HeapItem {
    key: Vec<u8>,
    table_idx: usize,
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .key
            .cmp(&self.key)
            .then_with(|| other.table_idx.cmp(&self.table_idx))
    }
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct MergeIter<'a> {
    pub iters: Vec<TableIter<'a>>,
    heap: BinaryHeap<HeapItem>,
}

impl<'a> MergeIter<'a> {
    pub fn new(tables: &'a [Vec<u8>]) -> Result<Self, String> {
        let mut iters = Vec::with_capacity(tables.len());
        let mut heap = BinaryHeap::new();
        for (i, table) in tables.iter().enumerate() {
            let iter = TableIter::new(table)?;
            if let Some(rec) = iter.current() {
                heap.push(HeapItem { key: rec.key.clone(), table_idx: i });
            }
            iters.push(iter);
        }
        Ok(Self { iters, heap })
    }

    pub fn next(&mut self) -> Result<Option<Record>, String> {
        let Some(item) = self.heap.pop() else {
            return Ok(None);
        };
        let iter = &mut self.iters[item.table_idx];
        let rec = iter.current().unwrap().clone();
        iter.advance()?;
        if let Some(next_rec) = iter.current() {
            self.heap.push(HeapItem { key: next_rec.key.clone(), table_idx: item.table_idx });
        }
        Ok(Some(rec))
    }
}
