import type { Manifest } from "../types";

interface Props { manifest: Manifest; idx: number; onSelect: (i: number) => void; }

export function FrameBrowser({ manifest, idx, onSelect }: Props) {
  const rows = manifest.frames;
  const cur = rows.findIndex((f) => f.idx === idx);
  const step = (d: number) => {
    const n = Math.max(0, Math.min(rows.length - 1, cur + d));
    onSelect(rows[n].idx);
  };
  return (
    <div className="panel">
      <h3>Frames — {manifest.run_name}</h3>
      <div className="stepper">
        <button onClick={() => step(-1)} disabled={cur <= 0}>◀</button>
        <span>{cur + 1} / {rows.length}</span>
        <button onClick={() => step(1)} disabled={cur >= rows.length - 1}>▶</button>
      </div>
      <div className="frame-list">
        {rows.map((f) => (
          <div
            key={f.idx}
            className={"frame-row" + (f.idx === idx ? " active" : "")}
            onClick={() => onSelect(f.idx)}
          >
            <span className="fid">#{f.idx}</span>
            <span>{f.n_peaks} pk</span>
            <span className={f.success ? "ok" : "no"}>{f.n_refl} refl</span>
          </div>
        ))}
      </div>
    </div>
  );
}
