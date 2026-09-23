//! Chunk index: `(start, end, n_crystals, n_obs_upper)` per
//! ``----- Begin chunk -----`` block, matching the Python ``_index_chunks``.
//!
//! A chunk marker always ends a reflection table, so the scan state at every
//! ``----- Begin chunk -----`` line is the same whatever came before. That is
//! what lets the threaded scan split the file at arbitrary chunk starts and
//! still return exactly the serial result, also for truncated tables.

use std::io;
use std::path::Path;

use rayon::prelude::*;

use crate::lines::LineReader;
use crate::pystr::bytes_space;

pub const BEGIN_CHUNK: &[u8] = b"----- Begin chunk -----";
pub const END_CHUNK: &[u8] = b"----- End chunk -----";
const BEGIN_CRYSTAL: &[u8] = b"--- Begin crystal";
const REFL_BEGIN: &[u8] = b"Reflections measured after indexing";
const REFL_END: &[u8] = b"End of reflections";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chunk {
    pub start: u64,
    pub end: u64,
    pub n_crystals: i64,
    pub n_obs: i64,
}

#[inline]
pub fn is_chunk_marker(line: &[u8]) -> bool {
    line.starts_with(BEGIN_CHUNK) || line.starts_with(END_CHUNK)
}

/// ``len(line.split()) >= 10`` on raw bytes.
#[inline]
fn has_ten_fields(line: &[u8]) -> bool {
    let mut n = 0;
    let mut in_tok = false;
    for &b in line {
        let sp = bytes_space(b);
        if !sp && !in_tok {
            n += 1;
            if n == 10 {
                return true;
            }
        }
        in_tok = !sp;
    }
    false
}

struct Scan {
    chunks: Vec<Chunk>,
    /// The last chunk of the range, whose end is the caller's to set.
    open: Option<Chunk>,
}

/// Scan `[start, end)`. With `max_crystals`, stop once the flushed chunks hold
/// at least that many crystals (the chunk that crosses it is included).
fn scan(path: &Path, start: u64, end: u64, max_crystals: Option<i64>) -> io::Result<Scan> {
    let mut r = LineReader::open(path)?;
    r.set_range(start, end)?;
    let mut chunks = Vec::new();
    let mut open: Option<Chunk> = None;
    let mut cum = 0i64;
    let mut in_refl = false;
    let mut header_pending = false;

    while let Some((pos, line)) = r.next_line()? {
        if in_refl {
            if header_pending {
                header_pending = false;
                if !is_chunk_marker(line) {
                    continue;
                }
            }
            if line.starts_with(REFL_END) {
                in_refl = false;
                continue;
            }
            if !is_chunk_marker(line) {
                if has_ten_fields(line) {
                    if let Some(c) = open.as_mut() {
                        c.n_obs += 1;
                    }
                }
                continue;
            }
            in_refl = false;
        }
        match line.first() {
            Some(b'-') => {
                if line.starts_with(BEGIN_CHUNK) {
                    if let Some(mut c) = open.take() {
                        c.end = pos;
                        cum += c.n_crystals;
                        chunks.push(c);
                        if max_crystals.is_some_and(|m| cum >= m) {
                            return Ok(Scan { chunks, open: None });
                        }
                    }
                    open = Some(Chunk { start: pos, end: 0, n_crystals: 0, n_obs: 0 });
                } else if line.starts_with(BEGIN_CRYSTAL) {
                    // Crystals before the first chunk (header region) are not counted.
                    if let Some(c) = open.as_mut() {
                        c.n_crystals += 1;
                    }
                }
            }
            Some(b'R') if line.starts_with(REFL_BEGIN) => {
                in_refl = true;
                header_pending = true;
            }
            _ => {}
        }
    }
    Ok(Scan { chunks, open })
}

/// Offset of the first ``----- Begin chunk -----`` line starting at or after
/// the line that contains `target` (the partial line at `target` is skipped),
/// or the file size.
fn find_chunk_at_or_after(path: &Path, target: u64, size: u64) -> io::Result<u64> {
    let mut r = LineReader::open(path)?;
    r.set_range(target.min(size), size)?;
    if target > 0 && r.next_line()?.is_none() {
        return Ok(size);
    }
    while let Some((pos, line)) = r.next_line()? {
        if line.starts_with(BEGIN_CHUNK) {
            return Ok(pos);
        }
    }
    Ok(size)
}

pub fn index_chunks(path: &Path, n_threads: usize, max_crystals: Option<i64>) -> io::Result<Vec<Chunk>> {
    let size = std::fs::metadata(path)?.len();
    let finish = |s: Scan, end: u64| {
        let mut v = s.chunks;
        if let Some(mut c) = s.open {
            c.end = end;
            v.push(c);
        }
        v
    };
    if max_crystals.is_some() || n_threads <= 1 {
        return Ok(finish(scan(path, 0, size, max_crystals)?, size));
    }

    let first = find_chunk_at_or_after(path, 0, size)?;
    if first >= size {
        return Ok(Vec::new());
    }
    let mut starts = vec![first];
    for i in 1..n_threads as u64 {
        let target = first + (size - first) * i / n_threads as u64;
        let s = find_chunk_at_or_after(path, target, size)?;
        if s > *starts.last().unwrap() && s < size {
            starts.push(s);
        }
    }
    let mut bounds: Vec<(u64, u64)> = starts.windows(2).map(|w| (w[0], w[1])).collect();
    bounds.push((*starts.last().unwrap(), size));

    let scans: Vec<Scan> = bounds
        .par_iter()
        .map(|&(s, e)| scan(path, s, e, None))
        .collect::<io::Result<_>>()?;
    let mut out = Vec::new();
    for (scan, &(_, e)) in scans.into_iter().zip(&bounds) {
        out.extend(finish(scan, e));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ten_fields() {
        assert!(has_ten_fields(b"   1    2    3 1.0 2.0 3.0 4.0 5.0 6.0 p0"));
        assert!(!has_ten_fields(b"   1    2    3 1.0 2.0 3.0 4.0 5.0 6.0"));
        assert!(!has_ten_fields(b""));
    }
}
