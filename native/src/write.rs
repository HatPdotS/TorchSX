//! CrystFEL ``.stream`` writer, byte-identical to the Python
//! ``_write_stream_reconstruct_python`` for cores without a ``peak`` level.
//!
//! Numbers follow Python's ``format``: values are f32 widened exactly to f64,
//! the unit conversions the Python writer does on float32 arrays are done in
//! f32, NaN prints as ``nan`` (``+nan`` with a sign flag), and ties round to
//! even on the exact binary value (Rust's and Python's float formatting agree
//! on that).
//!
//! Images are formatted in parallel in batches and written in order, so memory
//! stays at about one batch of text whatever the output size.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use rayon::prelude::*;

pub struct WriteInput<'a> {
    pub hkl: &'a [i16],
    pub i: &'a [f32],
    pub sigma: &'a [f32],
    pub peak: &'a [f32],
    pub background: &'a [f32],
    pub fs: &'a [f32],
    pub ss: &'a [f32],
    pub panel_id: &'a [i16],
    pub cell: &'a [f32],
    pub astar: &'a [f32],
    pub bstar: &'a [f32],
    pub cstar: &'a [f32],
    pub profile_rad: &'a [f32],
    pub resolution: &'a [f32],
    pub n_refl: &'a [i32],
    pub image_id: &'a [i32],
    pub beam_energy: &'a [f32],
    pub n_peaks: &'a [i32],
    pub det_shift_x: &'a [f32],
    pub det_shift_y: &'a [f32],
    pub camera_length: &'a [f32],
    pub filenames: &'a [String],
    pub events: &'a [String],
    pub panel_names: &'a [String],
    pub lattice_type: Option<&'a str>,
    pub centering: Option<&'a str>,
    pub unique_axis: Option<&'a str>,
}

fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, msg)
}

struct Layout {
    obs_off: Vec<usize>,
    image_to_crystals: Vec<Vec<usize>>,
    /// Panel name index per observation (Python list indexing, negatives wrap).
    pname: Vec<usize>,
}

fn layout(w: &WriteInput) -> io::Result<Layout> {
    let n_obs = w.i.len();
    let n_crys = w.n_refl.len();
    let n_img = w.beam_energy.len();
    let check = |name: &str, len: usize, want: usize| {
        if len == want {
            Ok(())
        } else {
            Err(invalid(format!("{name}: length {len}, expected {want}")))
        }
    };
    check("hkl", w.hkl.len(), 3 * n_obs)?;
    for (name, a) in [("sigma", w.sigma), ("peak", w.peak), ("background", w.background), ("fs", w.fs), ("ss", w.ss)] {
        check(name, a.len(), n_obs)?;
    }
    check("panel_id", w.panel_id.len(), n_obs)?;
    check("cell", w.cell.len(), 6 * n_crys)?;
    for (name, a) in [("astar", w.astar), ("bstar", w.bstar), ("cstar", w.cstar)] {
        check(name, a.len(), 3 * n_crys)?;
    }
    check("profile_rad", w.profile_rad.len(), n_crys)?;
    check("resolution", w.resolution.len(), n_crys)?;
    check("image_id", w.image_id.len(), n_crys)?;
    check("n_peaks", w.n_peaks.len(), n_img)?;
    check("det_shift_x", w.det_shift_x.len(), n_img)?;
    check("det_shift_y", w.det_shift_y.len(), n_img)?;

    let mut obs_off = Vec::with_capacity(n_crys + 1);
    obs_off.push(0usize);
    for (cid, &n) in w.n_refl.iter().enumerate() {
        let next = usize::try_from(n).ok().map(|n| obs_off[cid] + n).filter(|&e| e <= n_obs);
        obs_off.push(next.ok_or_else(|| invalid(format!("n_refl[{cid}] = {n} overruns the {n_obs} observations")))?);
    }
    let mut image_to_crystals = vec![Vec::new(); n_img];
    for (cid, &iid) in w.image_id.iter().enumerate() {
        if let Some(v) = usize::try_from(iid).ok().and_then(|i| image_to_crystals.get_mut(i)) {
            v.push(cid);
        }
    }
    let n_names = w.panel_names.len() as i64;
    let pname = w
        .panel_id
        .iter()
        .map(|&p| {
            let p = p as i64;
            let idx = if p < 0 { p + n_names } else { p };
            if (0..n_names).contains(&idx) {
                Ok(idx as usize)
            } else {
                Err(invalid(format!("panel_id {p} has no name ({n_names} panel names)")))
            }
        })
        .collect::<io::Result<_>>()?;
    Ok(Layout { obs_off, image_to_crystals, pname })
}

/// ``format(x, f"{w}.{p}f")``.
#[inline]
fn fixed(out: &mut Vec<u8>, x: f64, w: usize, p: usize) {
    if x.is_nan() {
        let _ = write!(out, "{:>w$}", "nan");
    } else {
        let _ = write!(out, "{x:w$.p$}");
    }
}

/// ``format(x, f"+.{p}f")``.
#[inline]
fn signed(out: &mut Vec<u8>, x: f64, p: usize) {
    if x.is_nan() {
        out.extend_from_slice(b"+nan");
    } else {
        let _ = write!(out, "{x:+.p$}");
    }
}

fn format_image(out: &mut Vec<u8>, iid: usize, w: &WriteInput, l: &Layout) {
    let crystals = &l.image_to_crystals[iid];
    out.extend_from_slice(b"----- Begin chunk -----\n");
    match w.filenames.get(iid) {
        Some(f) => {
            out.extend_from_slice(b"Image filename: ");
            out.extend_from_slice(f.as_bytes());
            out.push(b'\n');
        }
        None => {
            let _ = writeln!(out, "Image filename: image_{iid}");
        }
    }
    match w.events.get(iid) {
        Some(e) => {
            out.extend_from_slice(b"Event: ");
            out.extend_from_slice(e.as_bytes());
            out.push(b'\n');
        }
        None => {
            let _ = writeln!(out, "Event: //{iid}");
        }
    }
    let _ = writeln!(out, "Image serial number: {iid}");
    out.extend_from_slice(b"hit = 1\n");
    out.extend_from_slice(if crystals.is_empty() { b"indexed_by = none\n" } else { b"indexed_by = xgandalf\n" });
    out.extend_from_slice(b"n_indexing_tries = 1\n");
    out.extend_from_slice(b"photon_energy_eV = ");
    fixed(out, w.beam_energy[iid] as f64, 0, 6);
    out.extend_from_slice(b"\nbeam_divergence = 0.00e+00 rad\nbeam_bandwidth = 1.00e-08 (fraction)\n");
    out.extend_from_slice(b"average_camera_length = ");
    fixed(out, w.camera_length.get(iid).map_or(0.0, |&c| c as f64), 0, 6);
    let _ = writeln!(out, " m\nnum_peaks = {}", w.n_peaks[iid]);
    out.extend_from_slice(b"Peaks from peak search\n  fs/px   ss/px (1/d)/nm^-1   Intensity  Panel\nEnd of peak list\n");

    for &cid in crystals {
        out.extend_from_slice(b"--- Begin crystal\nCell parameters ");
        let cell = &w.cell[6 * cid..6 * cid + 6];
        for (k, &v) in cell.iter().enumerate() {
            let v = if k < 3 { (v / 10.0f32) as f64 } else { v as f64 };
            fixed(out, v, 0, 5);
            out.extend_from_slice(match k {
                2 => b" nm, ",
                5 => b" deg\n",
                _ => b" ",
            });
        }
        for (name, vec) in [("astar", w.astar), ("bstar", w.bstar), ("cstar", w.cstar)] {
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(b" =");
            for &v in &vec[3 * cid..3 * cid + 3] {
                out.push(b' ');
                signed(out, (v * 10.0f32) as f64, 7);
            }
            out.extend_from_slice(b" nm^-1\n");
        }
        for (key, val) in [("lattice_type", w.lattice_type), ("centering", w.centering), ("unique_axis", w.unique_axis)] {
            if let Some(val) = val {
                let _ = writeln!(out, "{key} = {val}");
            }
        }
        out.extend_from_slice(b"profile_radius = ");
        fixed(out, (w.profile_rad[cid] * 10.0f32) as f64, 0, 5);
        out.extend_from_slice(
            b" nm^-1\n\
predict_refine/initial_residual = 0.000000 (0.000000 0.000000 0.000000)\n\
predict_refine/final_residual = 0.000000 (0.000000 0.000000 0.000000)\n\
predict_refine/total_shifts = 0.000000e+00 0.000000e+00 0.000000e+00\n\
predict_refine/det_shift x = ",
        );
        fixed(out, w.det_shift_x[iid] as f64, 0, 3);
        out.extend_from_slice(b" y = ");
        fixed(out, w.det_shift_y[iid] as f64, 0, 3);
        out.extend_from_slice(b" mm\ndiffraction_resolution_limit = ");
        let res_a = w.resolution[cid] as f64;
        fixed(out, if res_a > 0.0 { 10.0 / res_a } else { 0.0 }, 0, 2);
        out.extend_from_slice(b" nm^-1 or ");
        fixed(out, res_a, 0, 2);
        let _ = writeln!(out, " A\nnum_reflections = {}", w.n_refl[cid]);
        out.extend_from_slice(
            b"num_saturated_reflections = 0\n\
num_implausible_reflections = 0\n\
Reflections measured after indexing\n   \
h    k    l          I   sigma(I)       peak background  fs/px  ss/px panel\n",
        );
        for j in l.obs_off[cid]..l.obs_off[cid + 1] {
            let h = &w.hkl[3 * j..3 * j + 3];
            let _ = write!(out, "{:4} {:4} {:4} ", h[0], h[1], h[2]);
            fixed(out, w.i[j] as f64, 10, 2);
            out.push(b' ');
            fixed(out, w.sigma[j] as f64, 10, 2);
            out.push(b' ');
            fixed(out, w.peak[j] as f64, 10, 2);
            out.push(b' ');
            fixed(out, w.background[j] as f64, 10, 2);
            fixed(out, w.fs[j] as f64, 7, 1);
            out.push(b' ');
            fixed(out, w.ss[j] as f64, 6, 1);
            out.push(b' ');
            out.extend_from_slice(w.panel_names[l.pname[j]].as_bytes());
            out.push(b'\n');
        }
        out.extend_from_slice(b"End of reflections\n--- End crystal\n");
    }
    out.extend_from_slice(b"----- End chunk -----\n");
}

pub fn write_stream(path: &Path, header: &[u8], w: &WriteInput, n_threads: usize, images_per_batch: usize) -> io::Result<()> {
    let l = layout(w)?;
    let n_img = w.beam_energy.len();
    let batch = images_per_batch.max(1);
    let mut f = BufWriter::with_capacity(1 << 20, File::create(path)?);
    f.write_all(header)?;
    let mut buf = Vec::with_capacity(1 << 20);
    for start in (0..n_img).step_by(batch) {
        let stop = (start + batch).min(n_img);
        if n_threads <= 1 {
            buf.clear();
            for iid in start..stop {
                format_image(&mut buf, iid, w, &l);
            }
            f.write_all(&buf)?;
        } else {
            let parts: Vec<Vec<u8>> = (start..stop)
                .into_par_iter()
                .map(|iid| {
                    let mut b = Vec::with_capacity(8 << 10);
                    format_image(&mut b, iid, w, &l);
                    b
                })
                .collect();
            for p in &parts {
                f.write_all(p)?;
            }
        }
    }
    f.flush()
}
