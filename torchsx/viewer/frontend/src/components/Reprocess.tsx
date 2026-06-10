// "Reprocess (live)" panel: re-run the per-frame pipeline with typed parameters
// and overlay the results. Preview only — the stream on disk is never modified.
//
//   ① Find    — re-run the bayes finder with top_n (= hits fed to the indexer) + σ
//   ② Index   — index the live peaks (else the stream's) against a typed seed cell;
//               on success it auto-fires Predict (per the agreed workflow)
//   ③ Predict — predict reflections from a (hand-editable) basis, with the headline
//               Ewald-offset cutoff offset_max (Å⁻¹) + d_min; independently re-runnable
//
// Find and Index are separate buttons; Index → Predict is automatic. The basis is
// editable so the prediction can be nudged by hand.

import { useEffect, useState } from "react";
import * as api from "../api";
import type { Crystal, FindResult, PredictResult, Solution } from "../types";

interface Props {
  idx: number | null;
  streamCrystal: Crystal | null;
  find: FindResult | null;
  setFind: (f: FindResult | null) => void;
  solution: Solution | null;
  setSolution: (s: Solution | null) => void;
  predicted: PredictResult | null;
  setPredicted: (p: PredictResult | null) => void;
}

const HEWL_CELL: number[] = [79.1, 79.1, 38.0, 90, 90, 90];

// real-space basis (rows a,b,c) from reciprocal rows a*,b*,c*: B = inv(B*)ᵀ (involutive).
function recipToBasis(astar: number[], bstar: number[], cstar: number[]): number[][] | null {
  const m = [astar, bstar, cstar];
  const inv = invert3(m);
  if (!inv) return null;
  return transpose3(inv);
}
function invert3(m: number[][]): number[][] | null {
  const a = m[0][0], b = m[0][1], c = m[0][2];
  const d = m[1][0], e = m[1][1], f = m[1][2];
  const g = m[2][0], h = m[2][1], i = m[2][2];
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-20) return null;
  const id = 1 / det;
  return [
    [A * id, (c * h - b * i) * id, (b * f - c * e) * id],
    [B * id, (a * i - c * g) * id, (c * d - a * f) * id],
    [C * id, (b * g - a * h) * id, (a * e - b * d) * id],
  ];
}
function transpose3(m: number[][]): number[][] {
  return [[m[0][0], m[1][0], m[2][0]], [m[0][1], m[1][1], m[2][1]], [m[0][2], m[1][2], m[2][2]]];
}

export function Reprocess(props: Props) {
  const { idx, streamCrystal, find, setFind, solution, setSolution, predicted, setPredicted } = props;

  // ① find inputs
  const [topN, setTopN] = useState(200);
  const [sr, setSr] = useState(2.0);
  const [sa, setSa] = useState(0.7);
  const [pmd, setPmd] = useState(2);
  // ② index inputs (seed cell)
  const [cell, setCell] = useState<number[]>(HEWL_CELL);
  // ③ predict inputs
  const [basis, setBasis] = useState<number[][]>([[79.1, 0, 0], [0, 79.1, 0], [0, 0, 38]]);
  const [dMin, setDMin] = useState(2.0);
  const [offset, setOffset] = useState(0.006);

  const [busy, setBusy] = useState<null | "find" | "index" | "predict">(null);
  const [err, setErr] = useState<string | null>(null);

  // Prefill seed cell + basis from the frame's stream crystal (if any) when the
  // frame changes; fall back to HEWL.
  useEffect(() => {
    if (streamCrystal) {
      setCell([...streamCrystal.cell_lengths, ...streamCrystal.cell_angles]);
      const b = recipToBasis(streamCrystal.astar, streamCrystal.bstar, streamCrystal.cstar);
      if (b) setBasis(b);
    } else {
      setCell(HEWL_CELL);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [idx, streamCrystal]);

  if (idx == null) return null;

  const runFind = async () => {
    setBusy("find"); setErr(null);
    try {
      setFind(await api.postFindSpots(idx, { top_n: topN, sigma_radial: sr, sigma_azimuthal: sa, peak_min_distance: pmd }));
    } catch (e) { setErr(String(e)); }
    finally { setBusy(null); }
  };

  const runPredict = async (b: number[][]) => {
    setBusy("predict"); setErr(null);
    try {
      setPredicted(await api.postPredict(idx, { basis: b, d_min: dMin, offset_max: offset }));
    } catch (e) { setErr(String(e)); }
    finally { setBusy(null); }
  };

  const runIndex = async () => {
    setBusy("index"); setErr(null);
    try {
      const sol = await api.postIndex(idx, { spots: find?.peaks, cell });
      setSolution(sol);
      if (sol.success) {
        setBasis(sol.basis);
        await runPredict(sol.basis);   // indexing auto-triggers predict
      }
    } catch (e) { setErr(String(e)); }
    finally { setBusy(null); }
  };

  const num = (v: number, set: (n: number) => void, step = 1, w = 64) => (
    <input type="number" value={v} step={step} style={{ width: w }}
      onChange={(e) => set(e.target.value === "" ? 0 : +e.target.value)} />
  );

  return (
    <div className="panel">
      <h3>Reprocess (live)</h3>
      {err && <p className="err">{err}</p>}

      {/* ① find */}
      <div className="reproc-step">
        <div className="row"><span>top_n (hits → indexer)</span>{num(topN, setTopN, 10, 72)}</div>
        <div className="row"><span>σ radial</span>{num(sr, setSr, 0.1)}</div>
        <div className="row"><span>σ azimuthal</span>{num(sa, setSa, 0.1)}</div>
        <div className="row"><span>peak min dist</span>{num(pmd, setPmd, 1)}</div>
        <button onClick={runFind} disabled={busy != null}>
          {busy === "find" ? "Finding… (calibrates on first run)" : "Find peaks"}
        </button>
        {find && (
          <p className="muted">found {find.n_found} · Σprob {find.sum_prob.toFixed(1)} · {find.hit ? "hit ✓" : "no hit ✗"}</p>
        )}
      </div>

      {/* ② index */}
      <div className="reproc-step">
        <div className="row"><span>seed cell a,b,c</span>
          {[0, 1, 2].map((k) => (
            <input key={k} type="number" value={cell[k]} step={0.5} style={{ width: 52 }}
              onChange={(e) => setCell(cell.map((x, j) => (j === k ? +e.target.value : x)))} />
          ))}
        </div>
        <div className="row"><span>α,β,γ</span>
          {[3, 4, 5].map((k) => (
            <input key={k} type="number" value={cell[k]} step={1} style={{ width: 52 }}
              onChange={(e) => setCell(cell.map((x, j) => (j === k ? +e.target.value : x)))} />
          ))}
        </div>
        <button onClick={runIndex} disabled={busy != null}>
          {busy === "index" ? "Indexing…" : `Index ${find ? "live peaks" : "stream peaks"} → predict`}
        </button>
        {solution && (
          <p className="muted">
            {solution.success
              ? `indexed ${solution.n_indexed} · cell ${solution.cell.slice(0, 3).map((x) => x.toFixed(1)).join(", ")} · err ${solution.max_error.toFixed(4)}`
              : `indexing failed (n_indexed ${solution.n_indexed})`}
          </p>
        )}
      </div>

      {/* ③ predict */}
      <div className="reproc-step">
        <div className="row"><span>Ewald offset (Å⁻¹)</span>{num(offset, setOffset, 0.001, 80)}</div>
        <div className="row"><span>d_min (Å)</span>{num(dMin, setDMin, 0.1)}</div>
        <details className="basis-edit">
          <summary>basis (rows a,b,c)</summary>
          {basis.map((rowv, ri) => (
            <div className="row" key={ri}>
              {rowv.map((v, ci) => (
                <input key={ci} type="number" value={v} step={0.1} style={{ width: 60 }}
                  onChange={(e) => setBasis(basis.map((rr, j) => (j === ri ? rr.map((x, c) => (c === ci ? +e.target.value : x)) : rr)))} />
              ))}
            </div>
          ))}
        </details>
        <button onClick={() => runPredict(basis)} disabled={busy != null}>
          {busy === "predict" ? "Predicting…" : "Predict"}
        </button>
        {predicted && <p className="muted">predicted {predicted.n_predicted} reflections</p>}
      </div>
    </div>
  );
}
