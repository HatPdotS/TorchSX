//! Python ``str`` semantics the parser has to reproduce exactly:
//! ``str.split()``/``str.strip()`` whitespace, ``int()`` and ``float()``.

use std::borrow::Cow;

/// ``str.isspace`` for one character: Unicode White_Space plus U+001C..U+001F,
/// which Python also treats as whitespace.
#[inline]
pub fn py_isspace(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

#[inline]
fn ascii_py_space(b: u8) -> bool {
    matches!(b, b'\t'..=b'\r' | 0x1c..=0x1f | b' ')
}

/// ``bytes.split()`` whitespace (used by the byte-level indexer).
#[inline]
pub fn bytes_space(b: u8) -> bool {
    matches!(b, b'\t'..=b'\r' | b' ')
}

pub fn py_strip(s: &str) -> &str {
    s.trim_matches(py_isspace)
}

/// ``str.split()`` with no arguments. ASCII lines, which is every line of a
/// real stream, take a byte-level path.
pub fn py_split(s: &str) -> PySplit<'_> {
    PySplit { s, pos: 0, ascii: s.is_ascii() }
}

pub struct PySplit<'a> {
    s: &'a str,
    pos: usize,
    ascii: bool,
}

impl<'a> Iterator for PySplit<'a> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<&'a str> {
        if self.ascii {
            let b = self.s.as_bytes();
            let mut i = self.pos;
            while i < b.len() && ascii_py_space(b[i]) {
                i += 1;
            }
            if i == b.len() {
                self.pos = i;
                return None;
            }
            let start = i;
            while i < b.len() && !ascii_py_space(b[i]) {
                i += 1;
            }
            self.pos = i;
            Some(&self.s[start..i])
        } else {
            let rest = &self.s[self.pos..];
            let start = rest.find(|c: char| !py_isspace(c))?;
            let tail = &rest[start..];
            let len = tail.find(py_isspace).unwrap_or(tail.len());
            self.pos += start + len;
            Some(&tail[..len])
        }
    }
}

/// Remove PEP 515 underscores (each must sit between two ASCII digits);
/// ``None`` if an underscore is misplaced.
fn strip_underscores(t: &str) -> Option<Cow<'_, str>> {
    if !t.contains('_') {
        return Some(Cow::Borrowed(t));
    }
    let b = t.as_bytes();
    for (i, &c) in b.iter().enumerate() {
        if c == b'_' && !(i > 0 && i + 1 < b.len() && b[i - 1].is_ascii_digit() && b[i + 1].is_ascii_digit()) {
            return None;
        }
    }
    Some(Cow::Owned(t.replace('_', "")))
}

pub enum PyInt {
    Ok(i64),
    /// A valid integer literal outside the i64 range (Python would accept it).
    Overflow,
    Invalid,
}

/// ``int(token)`` for a whitespace-free token.
pub fn py_int(t: &str) -> PyInt {
    let Some(t) = strip_underscores(t) else { return PyInt::Invalid };
    let b = t.as_bytes();
    let digits = match b.first() {
        Some(b'+' | b'-') => &b[1..],
        _ => b,
    };
    if digits.is_empty() || !digits.iter().all(u8::is_ascii_digit) {
        return PyInt::Invalid;
    }
    match t.parse::<i64>() {
        Ok(v) => PyInt::Ok(v),
        Err(_) => PyInt::Overflow,
    }
}

/// ``float(token)``: correctly rounded, accepts inf/infinity/nan in any case.
#[inline]
pub fn py_float(t: &str) -> Option<f64> {
    if !t.contains('_') {
        return t.parse::<f64>().ok();
    }
    strip_underscores(t)?.parse::<f64>().ok()
}

/// ``float(v)`` of a chunk value that was stored as ``int(v0)`` if possible,
/// else ``float(v0)``. Python raises for an integer too large for a float.
pub fn py_number_f64(t: &str) -> Option<f64> {
    match py_int(t) {
        PyInt::Ok(i) => Some(i as f64),
        PyInt::Overflow => py_float(t).filter(|f| f.is_finite()),
        PyInt::Invalid => py_float(t),
    }
}

/// ``int(v)`` of the same stored value, then an int32 store.
pub fn py_number_i32(t: &str) -> Option<i32> {
    let v = match py_int(t) {
        PyInt::Ok(i) => i,
        PyInt::Overflow => return None,
        PyInt::Invalid => {
            let f = py_float(t)?;
            if !f.is_finite() || f.trunc().abs() > i64::MAX as f64 {
                return None;
            }
            f.trunc() as i64
        }
    };
    i32::try_from(v).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_matches_python() {
        assert_eq!(py_split("  a \t b\x0bc\x1cd  ").collect::<Vec<_>>(), ["a", "b", "c", "d"]);
        assert_eq!(py_split("x\u{a0}y\u{2003}z").collect::<Vec<_>>(), ["x", "y", "z"]);
        assert_eq!(py_split("   ").count(), 0);
        assert_eq!(py_strip("\x1f a b \u{3000}"), "a b");
    }

    #[test]
    fn numbers_match_python() {
        assert!(matches!(py_int("+12"), PyInt::Ok(12)));
        assert!(matches!(py_int("1_000"), PyInt::Ok(1000)));
        assert!(matches!(py_int("1__0"), PyInt::Invalid));
        assert!(matches!(py_int("1.0"), PyInt::Invalid));
        assert!(matches!(py_int("99999999999999999999"), PyInt::Overflow));
        assert_eq!(py_float("1_0.5"), Some(10.5));
        assert_eq!(py_float("-Infinity"), Some(f64::NEG_INFINITY));
        assert!(py_float("nAn").unwrap().is_nan());
        assert_eq!(py_float("1e"), None);
        assert_eq!(py_float("nm,"), None);
        assert_eq!(py_number_i32("1e3"), Some(1000));
        assert_eq!(py_number_i32("-2.7"), Some(-2));
        assert_eq!(py_number_i32("3000000000"), None);
        assert_eq!(py_number_i32("nan"), None);
        assert_eq!(py_number_f64("12"), Some(12.0));
    }
}
