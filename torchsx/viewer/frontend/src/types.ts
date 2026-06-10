// Shapes mirror the viewer server (torchsx/viewer/server.py).

export interface Panel {
  name: string;
  min_fs: number; max_fs: number; min_ss: number; max_ss: number;
  corner_x: number; corner_y: number;
  fs_vec: [number, number, number];
  ss_vec: [number, number, number];
  coffset: number;
  n_ss: number; n_fs: number;
}

export interface Geometry {
  photon_energy_eV: number;
  wavelength_A: number;
  clen: number;
  res: number;
  panels: Panel[];
}

export interface FrameRow {
  idx: number;
  n_peaks: number;
  n_found: number;   // raw peak-finder hits retained in the stream's peak level
  n_refl: number;
  n_crystals: number;
  cell_lengths: number[] | null;
  success: boolean;
}

export interface Manifest {
  run_name: string;
  detector: { n_panels: number; n_ss: number; n_fs: number };
  frames: FrameRow[];
  score_default: { sigma_radial: number; sigma_azimuthal: number };
}

export interface CountsMeta { shape: [number, number, number]; p90: number; vmax: number; }

// An indexed reflection from the stream (carries hkl).
export interface Reflection {
  panel: number; ss: number; fs: number; intensity: number;
  prob: number | null; hkl: [number, number, number];
}

// A raw peak-finder hit (the stream's `peak` level) — a found spot, indexed or not.
export interface FoundPeak {
  panel: number; ss: number; fs: number; intensity: number; prob: number | null;
}

export interface Crystal {
  cell_lengths: number[]; cell_angles: number[];
  astar: number[]; bstar: number[]; cstar: number[];
  resolution: number; n_refl: number;
}

// Detector counts held in memory for pixel-value lookup + texture upload.
export interface Counts {
  data: Uint16Array;            // length P*n_ss*n_fs, C-order
  shape: [number, number, number];
  meta: CountsMeta;
}

// Matched-filter score map (raw float16 bits for the HALF_FLOAT texture).
export interface Score {
  data: Uint16Array;            // half-float bit pattern, length P*n_ss*n_fs
  shape: [number, number, number];
  smin: number; smax: number;
}

// Detector good-pixel mask (per-run constant). 1 = good, 0 = masked.
export interface Mask {
  data: Uint8Array;             // length P*n_ss*n_fs, C-order
  shape: [number, number, number];
}

// ── live reprocessing (find → index → predict), mirrors _reprocess.Reprocessor ──

// A live peak from a re-run of the finder (same fields as FoundPeak).
export type LivePeak = FoundPeak;

export interface FindResult {
  peaks: LivePeak[];
  n_found: number;
  sum_prob: number;
  hit: boolean;
}

// A re-indexed solution: refined real-space basis (rows a,b,c) + cell.
export interface Solution {
  success: boolean;
  n_indexed: number;
  max_error: number;
  basis: number[][];            // 3×3
  cell: number[];               // [a,b,c,α,β,γ]
}

// A predicted reflection (carries hkl + Ewald excitation error).
export interface PredReflection {
  panel: number; ss: number; fs: number;
  hkl: [number, number, number];
  excitation_error: number; d_spacing: number;
}

export interface PredictResult {
  reflections: PredReflection[];
  n_predicted: number;
}
