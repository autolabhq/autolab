use crate::block::Entry;
use crate::builder::TableBuilder;
use crate::merge::MergeIter;
use crate::types::{CompactionStats, Kind};

pub fn compact_tables(tables: &[Vec<u8>]) -> Result<CompactionStats, String> {
    let mut merge = MergeIter::new(tables)?;
    let mut builder = TableBuilder::new();
    let mut live = 0u64;
    let mut current_key: Option<[u8; 8]> = None;
    let mut winner: Option<Entry> = None;

    while let Some(rec) = merge.next()? {
        match current_key {
            None => {
                current_key = Some(rec.key);
                winner = Some(rec);
            }
            Some(key) if key == rec.key => {}
            Some(_) => {
                if let Some(w) = winner.take() {
                    if matches!(w.kind, Kind::Put) {
                        builder.add(&w);
                        live += 1;
                    }
                }
                current_key = Some(rec.key);
                winner = Some(rec);
            }
        }
    }

    if let Some(w) = winner.take() {
        if matches!(w.kind, Kind::Put) {
            builder.add(&w);
            live += 1;
        }
    }

    let mut stats = builder.finish();
    stats.live_entries = live;
    Ok(stats)
}
