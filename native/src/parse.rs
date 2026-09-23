//! Record parser: chunk byte ranges → column buffers.
//!
//! On well-formed streams the output is bit-identical to the Python
//! ``_parse_into_views`` (f64 parsing, unit arithmetic in f64, one cast to
//! f32). Malformed content is excluded at the smallest unit that keeps the rest
//! valid and counted by [`Reason`]; nothing here raises on bad input.
//!
//! A chunk marker (``----- Begin chunk -----`` or ``----- End chunk -----``)
//! ends whatever image or crystal is open, and ``--- Begin crystal`` starts a
//! crystal even where a peak section was expected. The state at each chunk start is
//! therefore fixed, so a range can be parsed in any split without changing the
//! result.

use std::collections::HashMap;
use std::io;
use std::path::Path;

use rayon::prelude::*;

use crate::index::{is_chunk_marker, BEGIN_CHUNK};
use crate::lines::LineReader;
use crate::pystr::{py_float, py_int, py_number_f64, py_number_i32, py_split, py_strip, PyInt};

#[derive(Clone, Copy)]
pub enum Reason {
    ReflectionShort,
    ReflectionValue,
    ReflectionHklRange,
    CrystalCell,
    CrystalVector,
    CrystalTruncated,
    ImageTruncated,
    ChunkNoImage,
    KeyValue,
    DetShift,
}

pub const REASON_NAMES: [&str; 10] = [
    "reflection_short",
    "reflection_value",
    "reflection_hkl_range",
    "crystal_cell",
    "crystal_vector",
    "crystal_truncated",
    "image_truncated",
    "chunk_no_image",
    "key_value",
    "det_shift",
];

/// Exclusions by reason: count and byte offset of the first occurrence.
#[derive(Clone)]
pub struct Report {
    pub count: [u64; 10],
    pub first: [u64; 10],
}

impl Report {
    fn new() -> Self {
        Self { count: [0; 10], first: [u64::MAX; 10] }
    }

    fn add(&mut self, r: Reason, off: u64) {
        let i = r as usize;
        self.count[i] += 1;
        self.first[i] = self.first[i].min(off);
    }

    fn merge(&mut self, o: &Report) {
        for i in 0..10 {
            self.count[i] += o.count[i];
            self.first[i] = self.first[i].min(o.first[i]);
        }
    }
}

/// Parsed columns; names, dtypes and flattened shapes of ``_allocate_tensors``.
#[derive(Default)]
pub struct Columns {
    pub hkl: Vec<i16>,
    pub i: Vec<f32>,
    pub sigma: Vec<f32>,
    pub peak: Vec<f32>,
    pub background: Vec<f32>,
    pub fs: Vec<f32>,
    pub ss: Vec<f32>,
    pub panel_id: Vec<i16>,
    pub crystal_id: Vec<i32>,
    pub cell: Vec<f32>,
    pub astar: Vec<f32>,
    pub bstar: Vec<f32>,
    pub cstar: Vec<f32>,
    pub profile_rad: Vec<f32>,
    pub resolution: Vec<f32>,
    pub n_refl: Vec<i32>,
    pub image_id: Vec<i32>,
    pub crystal_in_image: Vec<i32>,
    pub beam_energy: Vec<f32>,
    pub det_shift_x: Vec<f32>,
    pub det_shift_y: Vec<f32>,
    pub camera_length: Vec<f32>,
    pub n_peaks: Vec<i32>,
    pub filenames: Vec<String>,
    pub events: Vec<String>,
}

impl Columns {
    fn with_capacity(n_obs: usize, n_crys: usize, n_img: usize) -> Self {
        Self {
            hkl: Vec::with_capacity(3 * n_obs),
            i: Vec::with_capacity(n_obs),
            sigma: Vec::with_capacity(n_obs),
            peak: Vec::with_capacity(n_obs),
            background: Vec::with_capacity(n_obs),
            fs: Vec::with_capacity(n_obs),
            ss: Vec::with_capacity(n_obs),
            panel_id: Vec::with_capacity(n_obs),
            crystal_id: Vec::with_capacity(n_obs),
            cell: Vec::with_capacity(6 * n_crys),
            astar: Vec::with_capacity(3 * n_crys),
            bstar: Vec::with_capacity(3 * n_crys),
            cstar: Vec::with_capacity(3 * n_crys),
            profile_rad: Vec::with_capacity(n_crys),
            resolution: Vec::with_capacity(n_crys),
            n_refl: Vec::with_capacity(n_crys),
            image_id: Vec::with_capacity(n_crys),
            crystal_in_image: Vec::with_capacity(n_crys),
            beam_energy: Vec::with_capacity(n_img),
            det_shift_x: Vec::with_capacity(n_img),
            det_shift_y: Vec::with_capacity(n_img),
            camera_length: Vec::with_capacity(n_img),
            n_peaks: Vec::with_capacity(n_img),
            filenames: Vec::with_capacity(n_img),
            events: Vec::with_capacity(n_img),
        }
    }

    pub fn n_obs(&self) -> usize {
        self.i.len()
    }

    pub fn n_crystals(&self) -> usize {
        self.n_refl.len()
    }

    pub fn n_images(&self) -> usize {
        self.beam_energy.len()
    }

    fn truncate_obs(&mut self, n: usize) {
        self.hkl.truncate(3 * n);
        self.i.truncate(n);
        self.sigma.truncate(n);
        self.peak.truncate(n);
        self.background.truncate(n);
        self.fs.truncate(n);
        self.ss.truncate(n);
        self.panel_id.truncate(n);
        self.crystal_id.truncate(n);
    }
}

pub type PanelMap = HashMap<String, i16>;

type Kv<T> = Option<(Option<T>, u64)>;

#[derive(Clone, Copy)]
enum St {
    Outer,
    AfterFilename,
    ChunkKv,
    Peaks,
    Crystals,
    Cell,
    Vector(usize),
    CrystalKv,
    ReflHeader,
    Refl,
    AfterCrystal,
    SkipCrystal,
}

struct Parser<'a> {
    c: &'a mut Columns,
    rep: &'a mut Report,
    panels: &'a PanelMap,
    st: St,
    chunk_off: Option<u64>,
    chunk_has_image: bool,
    filename: String,
    filename_off: u64,
    event: String,
    energy: Kv<f64>,
    npeaks: Kv<i32>,
    clen: Kv<f64>,
    cur_image: usize,
    crystal_in_image: i32,
    crystal_off: u64,
    cell: [f64; 6],
    vecs: [[f64; 3]; 3],
    prof: Kv<f64>,
    res: Kv<f64>,
    det: Option<(f64, f64)>,
    obs_start: usize,
}

impl<'a> Parser<'a> {
    fn new(c: &'a mut Columns, rep: &'a mut Report, panels: &'a PanelMap) -> Self {
        Self {
            c,
            rep,
            panels,
            st: St::Outer,
            chunk_off: None,
            chunk_has_image: false,
            filename: String::new(),
            filename_off: 0,
            event: String::new(),
            energy: None,
            npeaks: None,
            clen: None,
            cur_image: 0,
            crystal_in_image: 0,
            crystal_off: 0,
            cell: [0.0; 6],
            vecs: [[0.0; 3]; 3],
            prof: None,
            res: None,
            det: None,
            obs_start: 0,
        }
    }

    fn feed(&mut self, off: u64, line: &str) {
        let marker = is_chunk_marker(line.as_bytes());
        loop {
            match self.st {
                St::Outer => {
                    if line.as_bytes().starts_with(BEGIN_CHUNK) {
                        self.close_chunk();
                        self.chunk_off = Some(off);
                        self.chunk_has_image = false;
                    } else if let Some(rest) = line.strip_prefix("Image filename:") {
                        self.chunk_has_image = true;
                        self.filename = py_strip(rest).to_owned();
                        self.filename_off = off;
                        self.st = St::AfterFilename;
                    }
                    return;
                }
                St::AfterFilename => {
                    if marker {
                        self.rep.add(Reason::ImageTruncated, self.filename_off);
                        self.st = St::Outer;
                        continue;
                    }
                    self.event = match line.strip_prefix("Event:") {
                        Some(rest) => py_strip(rest).to_owned(),
                        None => String::new(),
                    };
                    self.energy = None;
                    self.npeaks = None;
                    self.clen = None;
                    self.st = St::ChunkKv;
                    return;
                }
                St::ChunkKv => {
                    if line.starts_with("Peaks from peak search") {
                        self.commit_image();
                        self.st = St::Peaks;
                        return;
                    }
                    if marker {
                        self.commit_image();
                        self.st = St::Outer;
                        continue;
                    }
                    if line.starts_with("--- Begin crystal") {
                        // No peak section: the crystals still count.
                        self.commit_image();
                        self.st = St::Crystals;
                        continue;
                    }
                    self.chunk_kv(off, line);
                    return;
                }
                St::Peaks => {
                    if marker {
                        self.st = St::Outer;
                        continue;
                    }
                    if line.starts_with("End of peak list") {
                        self.st = St::Crystals;
                    } else if line.starts_with("--- Begin crystal") {
                        self.st = St::Crystals;
                        continue;
                    }
                    return;
                }
                St::Crystals => {
                    if marker {
                        self.st = St::Outer;
                        continue;
                    }
                    if line.starts_with("--- Begin crystal") {
                        self.crystal_off = off;
                        self.st = St::Cell;
                    }
                    return;
                }
                St::Cell => {
                    if marker {
                        self.truncated();
                        continue;
                    }
                    let mut n = 0;
                    for v in py_split(line).filter_map(py_float) {
                        self.cell[n] = v;
                        n += 1;
                        if n == 6 {
                            break;
                        }
                    }
                    self.st = if n == 6 {
                        St::Vector(0)
                    } else {
                        self.rep.add(Reason::CrystalCell, off);
                        St::SkipCrystal
                    };
                    return;
                }
                St::Vector(k) => {
                    if marker {
                        self.truncated();
                        continue;
                    }
                    let mut n = 0;
                    if let Some(eq) = line.find('=') {
                        for v in py_split(&line[eq + 1..]).filter_map(py_float) {
                            self.vecs[k][n] = v;
                            n += 1;
                            if n == 3 {
                                break;
                            }
                        }
                    }
                    self.st = if n < 3 {
                        self.rep.add(Reason::CrystalVector, off);
                        St::SkipCrystal
                    } else if k < 2 {
                        St::Vector(k + 1)
                    } else {
                        self.prof = None;
                        self.res = None;
                        self.det = None;
                        St::CrystalKv
                    };
                    return;
                }
                St::CrystalKv => {
                    if line.starts_with("Reflections measured after indexing") {
                        self.st = St::ReflHeader;
                        return;
                    }
                    if marker {
                        self.truncated();
                        continue;
                    }
                    self.crystal_kv(off, line);
                    return;
                }
                St::ReflHeader => {
                    if marker {
                        self.truncated();
                        continue;
                    }
                    self.obs_start = self.c.n_obs();
                    self.st = St::Refl;
                    return;
                }
                St::Refl => {
                    if line.starts_with("End of reflections") {
                        self.commit_crystal();
                        self.st = St::AfterCrystal;
                        return;
                    }
                    if marker {
                        self.truncated();
                        continue;
                    }
                    self.reflection(off, line);
                    return;
                }
                St::AfterCrystal | St::SkipCrystal => {
                    if marker {
                        self.st = St::Outer;
                        continue;
                    }
                    if line.starts_with("--- End crystal") {
                        self.st = St::Crystals;
                    }
                    return;
                }
            }
        }
    }

    /// End of the range: close whatever is open, as a chunk marker would.
    fn finish(&mut self) {
        match self.st {
            St::AfterFilename => self.rep.add(Reason::ImageTruncated, self.filename_off),
            St::ChunkKv => self.commit_image(),
            St::Cell | St::Vector(_) | St::CrystalKv | St::ReflHeader | St::Refl => self.truncated(),
            _ => {}
        }
        self.st = St::Outer;
        self.close_chunk();
    }

    fn close_chunk(&mut self) {
        if let Some(off) = self.chunk_off.take() {
            if !self.chunk_has_image {
                self.rep.add(Reason::ChunkNoImage, off);
            }
        }
    }

    /// Drop the open crystal and its reflections.
    fn truncated(&mut self) {
        if matches!(self.st, St::Refl) {
            let n = self.obs_start;
            self.c.truncate_obs(n);
        }
        self.rep.add(Reason::CrystalTruncated, self.crystal_off);
        self.st = St::Outer;
    }

    fn chunk_kv(&mut self, off: u64, line: &str) {
        let Some(eq) = line.find('=') else { return };
        let Some(v0) = py_split(&line[eq + 1..]).next() else { return };
        match py_strip(&line[..eq]) {
            "photon_energy_eV" => self.energy = Some((py_number_f64(v0), off)),
            "num_peaks" => self.npeaks = Some((py_number_i32(v0), off)),
            "average_camera_length" => self.clen = Some((py_number_f64(v0), off)),
            _ => {}
        }
    }

    fn crystal_kv(&mut self, off: u64, line: &str) {
        let Some(eq) = line.find('=') else { return };
        let key = py_strip(&line[..eq]);
        let val = py_strip(&line[eq + 1..]);
        if key.contains("det_shift") && val.contains("y =") {
            let mut parts = val.split("y =");
            let x = parts.next().and_then(|p| py_float(py_strip(p)));
            let y = parts.next().and_then(|p| py_float(py_strip(&py_strip(p).replace("mm", ""))));
            match (x, y) {
                (Some(x), Some(y)) => self.det = Some((x, y)),
                _ => self.rep.add(Reason::DetShift, off),
            }
            return;
        }
        let Some(first) = py_split(val).next() else { return };
        match key {
            "profile_radius" => self.prof = Some((py_float(first), off)),
            "diffraction_resolution_limit" => self.res = Some((py_float(first), off)),
            _ => {}
        }
    }

    fn commit_image(&mut self) {
        let energy = kv_or(self.rep, self.energy.take(), 0.0);
        let npeaks = kv_or(self.rep, self.npeaks.take(), 0);
        let clen = kv_or(self.rep, self.clen.take(), 0.0);
        let c = &mut *self.c;
        c.beam_energy.push(energy as f32);
        c.n_peaks.push(npeaks);
        c.det_shift_x.push(0.0);
        c.det_shift_y.push(0.0);
        c.camera_length.push(clen as f32);
        c.filenames.push(std::mem::take(&mut self.filename));
        c.events.push(std::mem::take(&mut self.event));
        self.cur_image = c.beam_energy.len() - 1;
        self.crystal_in_image = 0;
    }

    fn commit_crystal(&mut self) {
        let prof = kv_or(self.rep, self.prof.take(), 0.04);
        let res_nm = kv_or(self.rep, self.res.take(), 0.0);
        let c = &mut *self.c;
        let cell = &self.cell;
        c.cell.extend_from_slice(&[
            (cell[0] * 10.0) as f32,
            (cell[1] * 10.0) as f32,
            (cell[2] * 10.0) as f32,
            cell[3] as f32,
            cell[4] as f32,
            cell[5] as f32,
        ]);
        for (dst, v) in [&mut c.astar, &mut c.bstar, &mut c.cstar].into_iter().zip(&self.vecs) {
            dst.extend_from_slice(&[(v[0] / 10.0) as f32, (v[1] / 10.0) as f32, (v[2] / 10.0) as f32]);
        }
        c.profile_rad.push((prof / 10.0) as f32);
        c.resolution.push(if res_nm > 0.0 { 10.0 / res_nm } else { 999.0 } as f32);
        c.n_refl.push((c.i.len() - self.obs_start) as i32);
        c.image_id.push(self.cur_image as i32);
        c.crystal_in_image.push(self.crystal_in_image);
        self.crystal_in_image += 1;
        if let Some((x, y)) = self.det {
            c.det_shift_x[self.cur_image] = x as f32;
            c.det_shift_y[self.cur_image] = y as f32;
        }
    }

    fn reflection(&mut self, off: u64, line: &str) {
        let mut t = [""; 10];
        let mut n = 0;
        for tok in py_split(line) {
            t[n] = tok;
            n += 1;
            if n == 10 {
                break;
            }
        }
        if n < 10 {
            self.rep.add(Reason::ReflectionShort, off);
            return;
        }
        let mut hkl = [0i64; 3];
        let mut overflow = false;
        for (dst, tok) in hkl.iter_mut().zip(&t[..3]) {
            match py_int(tok) {
                PyInt::Ok(v) => *dst = v,
                PyInt::Overflow => overflow = true,
                PyInt::Invalid => {
                    self.rep.add(Reason::ReflectionValue, off);
                    return;
                }
            }
        }
        let mut vals = [0f64; 6];
        for (dst, tok) in vals.iter_mut().zip(&t[3..9]) {
            match py_float(tok) {
                Some(v) => *dst = v,
                None => {
                    self.rep.add(Reason::ReflectionValue, off);
                    return;
                }
            }
        }
        if overflow || hkl.iter().any(|&v| i16::try_from(v).is_err()) {
            self.rep.add(Reason::ReflectionHklRange, off);
            return;
        }
        let pid = self.panels.get(t[9]).copied().unwrap_or(0);
        let c = &mut *self.c;
        c.hkl.extend_from_slice(&[hkl[0] as i16, hkl[1] as i16, hkl[2] as i16]);
        c.i.push(vals[0] as f32);
        c.sigma.push(vals[1] as f32);
        c.peak.push(vals[2] as f32);
        c.background.push(vals[3] as f32);
        c.fs.push(vals[4] as f32);
        c.ss.push(vals[5] as f32);
        c.panel_id.push(pid);
        c.crystal_id.push(c.n_refl.len() as i32);
    }
}

/// A read key's value; a faulty value is reported and replaced by `default`.
fn kv_or<T>(rep: &mut Report, kv: Kv<T>, default: T) -> T {
    match kv {
        None => default,
        Some((Some(v), _)) => v,
        Some((None, off)) => {
            rep.add(Reason::KeyValue, off);
            default
        }
    }
}

/// Parse one work unit: a list of byte ranges, each starting at a chunk.
fn parse_unit(path: &Path, ranges: &[(u64, u64)], panels: &PanelMap, caps: (usize, usize, usize)) -> io::Result<(Columns, Report)> {
    let mut cols = Columns::with_capacity(caps.0, caps.1, caps.2);
    let mut rep = Report::new();
    let mut reader = LineReader::open(path)?;
    for &(s, e) in ranges {
        reader.set_range(s, e)?;
        let mut p = Parser::new(&mut cols, &mut rep, panels);
        while let Some((off, raw)) = reader.next_line()? {
            p.feed(off, &String::from_utf8_lossy(raw));
        }
        p.finish();
    }
    Ok((cols, rep))
}

/// Contiguous runs of chunks, balanced by observation count, with adjacent
/// byte ranges merged into one read.
fn work_units(chunks: &[[i64; 4]], n_units: usize) -> Vec<(Vec<(u64, u64)>, (usize, usize, usize))> {
    let total: i64 = chunks.iter().map(|c| c[3].max(1)).sum();
    let target = (total as f64 / n_units.max(1) as f64).max(1.0);
    let mut units = Vec::new();
    let mut cur: Vec<(u64, u64)> = Vec::new();
    let mut caps = (0usize, 0usize, 0usize);
    let mut acc = 0f64;
    for (i, c) in chunks.iter().enumerate() {
        let (s, e) = (c[0] as u64, c[1] as u64);
        match cur.last_mut() {
            Some(last) if last.1 == s => last.1 = e,
            _ => cur.push((s, e)),
        }
        caps.0 += c[3].max(0) as usize;
        caps.1 += c[2].max(0) as usize;
        caps.2 += 1;
        acc += c[3].max(1) as f64;
        if acc >= target && units.len() + 1 < n_units && i + 1 < chunks.len() {
            units.push((std::mem::take(&mut cur), caps));
            caps = (0, 0, 0);
            acc = 0.0;
        }
    }
    if !cur.is_empty() {
        units.push((cur, caps));
    }
    units
}

/// Concatenate unit columns in order, shifting the local crystal and image ids.
fn concat(mut parts: Vec<Columns>) -> Columns {
    if parts.len() == 1 {
        return parts.pop().unwrap();
    }
    let mut out = Columns::default();
    macro_rules! cat {
        ($($f:ident),*) => {$(
            let n: usize = parts.iter().map(|p| p.$f.len()).sum();
            let mut v = Vec::with_capacity(n);
            for p in parts.iter_mut() {
                v.append(&mut p.$f);
                p.$f = Vec::new();
            }
            out.$f = v;
        )*};
    }
    let (mut crys_base, mut img_base) = (0i32, 0i32);
    for p in parts.iter_mut() {
        if crys_base != 0 {
            p.crystal_id.iter_mut().for_each(|x| *x += crys_base);
        }
        if img_base != 0 {
            p.image_id.iter_mut().for_each(|x| *x += img_base);
        }
        crys_base += p.n_crystals() as i32;
        img_base += p.n_images() as i32;
    }
    cat!(hkl, i, sigma, peak, background, fs, ss, panel_id, crystal_id,
         cell, astar, bstar, cstar, profile_rad, resolution, n_refl, image_id, crystal_in_image,
         beam_energy, det_shift_x, det_shift_y, camera_length, n_peaks, filenames, events);
    out
}

/// Parse the given chunks (rows of `index_chunks`, file order) on `n_threads`.
pub fn parse_chunks(path: &Path, chunks: &[[i64; 4]], panels: &PanelMap, n_threads: usize) -> io::Result<(Columns, Report)> {
    let n_units = if n_threads <= 1 { 1 } else { (n_threads * 4).min(chunks.len().max(1)) };
    let units = work_units(chunks, n_units);
    let results: Vec<(Columns, Report)> = if units.len() <= 1 {
        units.iter().map(|(r, caps)| parse_unit(path, r, panels, *caps)).collect::<io::Result<_>>()?
    } else {
        units.par_iter().map(|(r, caps)| parse_unit(path, r, panels, *caps)).collect::<io::Result<_>>()?
    };
    let mut rep = Report::new();
    let mut parts = Vec::with_capacity(results.len());
    for (c, r) in results {
        rep.merge(&r);
        parts.push(c);
    }
    if parts.is_empty() {
        parts.push(Columns::default());
    }
    Ok((concat(parts), rep))
}
