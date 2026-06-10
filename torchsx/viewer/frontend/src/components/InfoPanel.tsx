import type { HoverInfo } from "./DetectorView";
import type { Crystal } from "../types";

interface Props { crystals: Crystal[]; nPeaks: number | null; hover: HoverInfo | null; }

export function InfoPanel({ crystals, nPeaks, hover }: Props) {
  const c = crystals[0];
  return (
    <div className="panel">
      <h3>Crystal</h3>
      {c ? (
        <div className="kv">
          <div><span>cell (Å)</span><b>{c.cell_lengths.map((x) => x.toFixed(1)).join(", ")}</b></div>
          <div><span>angles (°)</span><b>{c.cell_angles.map((x) => x.toFixed(1)).join(", ")}</b></div>
          <div><span>reflections</span><b>{c.n_refl}</b></div>
          <div><span>resolution (Å)</span><b>{c.resolution.toFixed(2)}</b></div>
          {crystals.length > 1 && <div><span>crystals</span><b>{crystals.length}</b></div>}
        </div>
      ) : <p className="muted">no indexed crystal</p>}
      {nPeaks != null && <div className="kv"><div><span>peaks found</span><b>{nPeaks}</b></div></div>}

      <h3>Cursor</h3>
      {hover ? (
        <div className="kv">
          <div><span>panel</span><b>{hover.panel}</b></div>
          <div><span>ss, fs</span><b>{hover.ss}, {hover.fs}</b></div>
          <div><span>counts</span><b>{hover.masked ? `[${hover.value}]` : hover.value}</b></div>
          {hover.masked && <div><span>pixel</span><b className="no">masked</b></div>}
          {hover.score != null && <div><span>score</span><b>{hover.score.toFixed(2)}</b></div>}
          {hover.pred && <div><span>pred hkl</span><b>{hover.pred.hkl.join(" ")}</b></div>}
          {hover.pred && <div><span>excit. err (Å⁻¹)</span><b>{hover.pred.excitation_error.toFixed(4)}</b></div>}
        </div>
      ) : <p className="muted">hover the detector</p>}
    </div>
  );
}
