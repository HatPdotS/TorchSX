//! torchsx_native: CrystFEL `.stream` index, parse and write for torchsx.
//!
//! The Python boundary is numpy arrays in and out; nothing here links torch.
//! Every call releases the GIL and runs on a pool of `n_threads` (1 = the
//! calling thread only). Bump [`PROTOCOL`] whenever a signature or a returned
//! layout changes; ``torchsx/io/_native.py`` refuses a mismatching build.

use std::path::PathBuf;

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

mod index;
mod lines;
mod parse;
mod pystr;
mod write;

const PROTOCOL: u32 = 1;

/// Invalid arguments become ValueError; everything else stays an OSError.
fn io_err(e: std::io::Error) -> PyErr {
    if e.kind() == std::io::ErrorKind::InvalidInput {
        PyValueError::new_err(e.to_string())
    } else {
        e.into()
    }
}

fn in_pool<R: Send>(n_threads: usize, f: impl FnOnce() -> R + Send) -> PyResult<R> {
    if n_threads <= 1 {
        return Ok(f());
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(pool.install(f))
}

/// Index the chunks of a stream: an ``(n, 4)`` int64 array of
/// ``(start, end, n_crystals, n_obs_upper)`` in file order. With
/// ``max_crystals``, stop after the chunk that brings the total to at least
/// that many crystals (serial).
#[pyfunction]
#[pyo3(signature = (path, n_threads=1, max_crystals=None))]
fn index_chunks<'py>(
    py: Python<'py>,
    path: PathBuf,
    n_threads: usize,
    max_crystals: Option<i64>,
) -> PyResult<Bound<'py, PyAny>> {
    let chunks = py.detach(|| in_pool(n_threads, || index::index_chunks(&path, n_threads, max_crystals)))?.map_err(io_err)?;
    let n = chunks.len();
    let flat: Vec<i64> = chunks
        .iter()
        .flat_map(|c| [c.start as i64, c.end as i64, c.n_crystals, c.n_obs])
        .collect();
    Ok(PyArray1::from_vec(py, flat).reshape([n, 4])?.into_any())
}

/// Parse chunk rows (from ``index_chunks``, file order) into columns.
///
/// Returns ``(columns, filenames, events, report)``: columns is a dict of
/// numpy arrays named as in ``_allocate_tensors``; report maps each exclusion
/// reason that occurred to ``(count, first_byte_offset)``.
#[pyfunction]
#[pyo3(signature = (path, chunks, panel_names, n_threads=1))]
fn parse_chunks<'py>(
    py: Python<'py>,
    path: PathBuf,
    chunks: PyReadonlyArray2<'py, i64>,
    panel_names: Vec<String>,
    n_threads: usize,
) -> PyResult<(Bound<'py, PyDict>, Vec<String>, Vec<String>, Bound<'py, PyDict>)> {
    let arr = chunks.as_array();
    if arr.ncols() != 4 {
        return Err(PyValueError::new_err(format!("chunks must have 4 columns, got {}", arr.ncols())));
    }
    let rows: Vec<[i64; 4]> = arr.rows().into_iter().map(|r| [r[0], r[1], r[2], r[3]]).collect();
    if let Some(bad) = rows.iter().find(|r| r[0] < 0 || r[1] < r[0]) {
        return Err(PyValueError::new_err(format!("invalid chunk byte range {:?}", &bad[..2])));
    }
    if panel_names.len() > i16::MAX as usize + 1 {
        return Err(PyValueError::new_err("more panels than an int16 panel_id can hold"));
    }
    let panels: parse::PanelMap = panel_names.into_iter().enumerate().map(|(i, n)| (n, i as i16)).collect();

    let (c, rep) = py.detach(|| in_pool(n_threads, || parse::parse_chunks(&path, &rows, &panels, n_threads)))?.map_err(io_err)?;
    let (n_obs, n_crys) = (c.n_obs(), c.n_crystals());
    let d = PyDict::new(py);
    d.set_item("hkl", PyArray1::from_vec(py, c.hkl).reshape([n_obs, 3])?)?;
    d.set_item("I", PyArray1::from_vec(py, c.i))?;
    d.set_item("sigma", PyArray1::from_vec(py, c.sigma))?;
    d.set_item("peak", PyArray1::from_vec(py, c.peak))?;
    d.set_item("background", PyArray1::from_vec(py, c.background))?;
    d.set_item("fs", PyArray1::from_vec(py, c.fs))?;
    d.set_item("ss", PyArray1::from_vec(py, c.ss))?;
    d.set_item("panel_id", PyArray1::from_vec(py, c.panel_id))?;
    d.set_item("crystal_id", PyArray1::from_vec(py, c.crystal_id))?;
    d.set_item("cell", PyArray1::from_vec(py, c.cell).reshape([n_crys, 6])?)?;
    d.set_item("astar", PyArray1::from_vec(py, c.astar).reshape([n_crys, 3])?)?;
    d.set_item("bstar", PyArray1::from_vec(py, c.bstar).reshape([n_crys, 3])?)?;
    d.set_item("cstar", PyArray1::from_vec(py, c.cstar).reshape([n_crys, 3])?)?;
    d.set_item("profile_rad", PyArray1::from_vec(py, c.profile_rad))?;
    d.set_item("resolution", PyArray1::from_vec(py, c.resolution))?;
    d.set_item("n_refl", PyArray1::from_vec(py, c.n_refl))?;
    d.set_item("image_id", PyArray1::from_vec(py, c.image_id))?;
    d.set_item("crystal_in_image", PyArray1::from_vec(py, c.crystal_in_image))?;
    d.set_item("beam_energy", PyArray1::from_vec(py, c.beam_energy))?;
    d.set_item("det_shift_x", PyArray1::from_vec(py, c.det_shift_x))?;
    d.set_item("det_shift_y", PyArray1::from_vec(py, c.det_shift_y))?;
    d.set_item("camera_length", PyArray1::from_vec(py, c.camera_length))?;
    d.set_item("n_peaks", PyArray1::from_vec(py, c.n_peaks))?;

    let report = PyDict::new(py);
    for (k, name) in parse::REASON_NAMES.iter().enumerate() {
        if rep.count[k] > 0 {
            report.set_item(*name, (rep.count[k], rep.first[k]))?;
        }
    }
    Ok((d, c.filenames, c.events, report))
}

fn array<'py, T: numpy::Element>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<PyReadonlyArrayDyn<'py, T>> {
    let v = d.get_item(key)?.ok_or_else(|| PyKeyError::new_err(key.to_owned()))?;
    v.extract::<PyReadonlyArrayDyn<'py, T>>()
        .map_err(|e| PyTypeError::new_err(format!("{key}: {e}")))
}

fn slice<'a, T: numpy::Element>(a: &'a PyReadonlyArrayDyn<'_, T>, key: &str) -> PyResult<&'a [T]> {
    a.as_slice().map_err(|_| PyValueError::new_err(format!("{key} must be C-contiguous")))
}

/// Write a ``.stream`` file: ``header`` verbatim, then one chunk per image.
///
/// ``arrays`` holds the observation (sorted by crystal), crystal and image
/// columns under their ``_allocate_tensors`` names; ``panel_names`` covers
/// every ``panel_id``; unit-cell fields that are ``None`` are omitted.
#[pyfunction]
#[pyo3(signature = (path, arrays, filenames, events, panel_names, lattice_type, centering, unique_axis, header, n_threads=1, images_per_batch=1024))]
#[allow(clippy::too_many_arguments)]
fn write_stream(
    py: Python<'_>,
    path: PathBuf,
    arrays: &Bound<'_, PyDict>,
    filenames: Vec<String>,
    events: Vec<String>,
    panel_names: Vec<String>,
    lattice_type: Option<String>,
    centering: Option<String>,
    unique_axis: Option<String>,
    header: String,
    n_threads: usize,
    images_per_batch: usize,
) -> PyResult<()> {
    let hkl = array::<i16>(arrays, "hkl")?;
    let i = array::<f32>(arrays, "I")?;
    let sigma = array::<f32>(arrays, "sigma")?;
    let peak = array::<f32>(arrays, "peak")?;
    let background = array::<f32>(arrays, "background")?;
    let fs = array::<f32>(arrays, "fs")?;
    let ss = array::<f32>(arrays, "ss")?;
    let panel_id = array::<i16>(arrays, "panel_id")?;
    let cell = array::<f32>(arrays, "cell")?;
    let astar = array::<f32>(arrays, "astar")?;
    let bstar = array::<f32>(arrays, "bstar")?;
    let cstar = array::<f32>(arrays, "cstar")?;
    let profile_rad = array::<f32>(arrays, "profile_rad")?;
    let resolution = array::<f32>(arrays, "resolution")?;
    let n_refl = array::<i32>(arrays, "n_refl")?;
    let image_id = array::<i32>(arrays, "image_id")?;
    let beam_energy = array::<f32>(arrays, "beam_energy")?;
    let n_peaks = array::<i32>(arrays, "n_peaks")?;
    let det_shift_x = array::<f32>(arrays, "det_shift_x")?;
    let det_shift_y = array::<f32>(arrays, "det_shift_y")?;
    let camera_length = array::<f32>(arrays, "camera_length")?;

    let w = write::WriteInput {
        hkl: slice(&hkl, "hkl")?,
        i: slice(&i, "I")?,
        sigma: slice(&sigma, "sigma")?,
        peak: slice(&peak, "peak")?,
        background: slice(&background, "background")?,
        fs: slice(&fs, "fs")?,
        ss: slice(&ss, "ss")?,
        panel_id: slice(&panel_id, "panel_id")?,
        cell: slice(&cell, "cell")?,
        astar: slice(&astar, "astar")?,
        bstar: slice(&bstar, "bstar")?,
        cstar: slice(&cstar, "cstar")?,
        profile_rad: slice(&profile_rad, "profile_rad")?,
        resolution: slice(&resolution, "resolution")?,
        n_refl: slice(&n_refl, "n_refl")?,
        image_id: slice(&image_id, "image_id")?,
        beam_energy: slice(&beam_energy, "beam_energy")?,
        n_peaks: slice(&n_peaks, "n_peaks")?,
        det_shift_x: slice(&det_shift_x, "det_shift_x")?,
        det_shift_y: slice(&det_shift_y, "det_shift_y")?,
        camera_length: slice(&camera_length, "camera_length")?,
        filenames: &filenames,
        events: &events,
        panel_names: &panel_names,
        lattice_type: lattice_type.as_deref(),
        centering: centering.as_deref(),
        unique_axis: unique_axis.as_deref(),
    };
    py.detach(|| in_pool(n_threads, || write::write_stream(&path, header.as_bytes(), &w, n_threads, images_per_batch)))?.map_err(io_err)?;
    Ok(())
}

#[pymodule]
fn torchsx_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("PROTOCOL", PROTOCOL)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(index_chunks, m)?)?;
    m.add_function(wrap_pyfunction!(parse_chunks, m)?)?;
    m.add_function(wrap_pyfunction!(write_stream, m)?)?;
    Ok(())
}
