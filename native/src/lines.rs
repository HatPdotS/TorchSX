//! Buffered line reader over a byte range of a file.
//!
//! Yields every line that *starts* inside `[start, end)`, like the Python
//! reader's ``while f.tell() < end: f.readline()``: a line that begins before
//! `end` is read to its newline even if that lies past `end`. Reads never run
//! more than one partial line past `end`, so scanning many small chunks (sampled
//! mode) reads only those chunks.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

const MAX_BUF: usize = 4 << 20;
const MIN_BUF: usize = 64 << 10;
/// Read step once past `end`, used only to finish the line that crosses it.
const OVERRUN_STEP: u64 = 64 << 10;

pub struct LineReader {
    file: File,
    buf: Vec<u8>,
    head: usize,
    tail: usize,
    /// File offset of `buf[0]`.
    buf_pos: u64,
    end: u64,
    eof: bool,
}

impl LineReader {
    pub fn open(path: &Path) -> io::Result<Self> {
        Ok(Self {
            file: File::open(path)?,
            buf: Vec::new(),
            head: 0,
            tail: 0,
            buf_pos: 0,
            end: 0,
            eof: false,
        })
    }

    /// Restart at `start`, yielding the lines that start before `end`.
    pub fn set_range(&mut self, start: u64, end: u64) -> io::Result<()> {
        self.file.seek(SeekFrom::Start(start))?;
        let want = (end.saturating_sub(start) as usize).clamp(MIN_BUF, MAX_BUF);
        if self.buf.len() < want {
            self.buf.resize(want, 0);
        }
        self.head = 0;
        self.tail = 0;
        self.buf_pos = start;
        self.end = end;
        self.eof = false;
        Ok(())
    }

    /// Next line as `(file offset, bytes)`, without the newline and without a
    /// trailing run of `\r`/`\n` (Python's ``rstrip(b"\r\n")``).
    pub fn next_line(&mut self) -> io::Result<Option<(u64, &[u8])>> {
        let off = self.buf_pos + self.head as u64;
        if off >= self.end {
            return Ok(None);
        }
        let mut searched = self.head;
        let (s, e, next_head) = loop {
            if let Some(i) = memchr::memchr(b'\n', &self.buf[searched..self.tail]) {
                let nl = searched + i;
                break (self.head, nl, nl + 1);
            }
            if self.eof {
                if self.head == self.tail {
                    return Ok(None);
                }
                break (self.head, self.tail, self.tail);
            }
            let scanned = self.tail - self.head;
            self.make_room();
            searched = scanned;
            self.fill()?;
        };
        self.head = next_head;
        let mut line = &self.buf[s..e];
        while let [rest @ .., b'\r' | b'\n'] = line {
            line = rest;
        }
        Ok(Some((off, line)))
    }

    /// Move the unread bytes to the front, doubling the buffer if a single
    /// line fills it.
    fn make_room(&mut self) {
        if self.head > 0 {
            self.buf.copy_within(self.head..self.tail, 0);
            self.tail -= self.head;
            self.buf_pos += self.head as u64;
            self.head = 0;
        }
        if self.tail == self.buf.len() {
            let n = (self.buf.len() * 2).max(MIN_BUF);
            self.buf.resize(n, 0);
        }
    }

    fn fill(&mut self) -> io::Result<()> {
        let file_pos = self.buf_pos + self.tail as u64;
        let limit = if file_pos < self.end { self.end } else { file_pos + OVERRUN_STEP };
        let n = ((limit - file_pos) as usize).min(self.buf.len() - self.tail);
        loop {
            match self.file.read(&mut self.buf[self.tail..self.tail + n]) {
                Ok(0) => {
                    self.eof = true;
                    return Ok(());
                }
                Ok(got) => {
                    self.tail += got;
                    return Ok(());
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn lines(content: &[u8], start: u64, end: u64) -> Vec<(u64, Vec<u8>)> {
        let dir = std::env::temp_dir().join(format!("tsxn-lines-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(format!("{start}-{end}-{}.txt", content.len()));
        std::fs::File::create(&p).unwrap().write_all(content).unwrap();
        let mut r = LineReader::open(&p).unwrap();
        r.set_range(start, end).unwrap();
        let mut out = Vec::new();
        while let Some((o, l)) = r.next_line().unwrap() {
            out.push((o, l.to_vec()));
        }
        out
    }

    #[test]
    fn crlf_and_missing_final_newline() {
        let got = lines(b"a\r\nbb\r\r\n\nccc", 0, 100);
        let want: Vec<(u64, Vec<u8>)> =
            vec![(0, b"a".to_vec()), (3, b"bb".to_vec()), (8, b"".to_vec()), (9, b"ccc".to_vec())];
        assert_eq!(got, want);
    }

    #[test]
    fn line_crossing_end_is_read_whole() {
        let got = lines(b"one\ntwo\nthree\n", 0, 5);
        assert_eq!(got, vec![(0, b"one".to_vec()), (4, b"two".to_vec())]);
        let got = lines(b"one\ntwo\nthree\n", 4, 8);
        assert_eq!(got, vec![(4, b"two".to_vec())]);
    }

    #[test]
    fn lines_longer_than_the_buffer() {
        let long = vec![b'x'; 3 * MIN_BUF + 17];
        let mut content = long.clone();
        content.extend_from_slice(b"\nend\n");
        let got = lines(&content, 0, content.len() as u64);
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].1, long);
        assert_eq!(got[1], (long.len() as u64 + 1, b"end".to_vec()));
    }
}
