import type { OverlayOpts } from "../overlay";
import type { CountsMeta } from "../types";

export interface ScoreControls {
  show: boolean; opacity: number; sigmaRadial: number; sigmaAzimuthal: number;
}

interface Props {
  meta: CountsMeta | null;
  contrast: { p90: number; vmax: number };
  setContrast: (c: { p90: number; vmax: number }) => void;
  overlay: OverlayOpts;
  setOverlay: (o: OverlayOpts) => void;
  score: ScoreControls;
  setScore: (s: ScoreControls) => void;
  scoreLoading: boolean;
}

export function Controls({ meta, contrast, setContrast, overlay, setOverlay, score, setScore, scoreLoading }: Props) {
  const hi = meta ? meta.vmax : 1000;
  return (
    <div className="panel">
      <h3>Overlays</h3>
      <label className="row">
        <input type="checkbox" checked={overlay.showPeaks}
          onChange={(e) => setOverlay({ ...overlay, showPeaks: e.target.checked })} />
        Found peaks (hits)
      </label>
      <label className="row">
        <input type="checkbox" checked={overlay.showResults}
          onChange={(e) => setOverlay({ ...overlay, showResults: e.target.checked })} />
        Indexed reflections
      </label>
      <label className="row">
        <input type="checkbox" checked={overlay.showLivePeaks}
          onChange={(e) => setOverlay({ ...overlay, showLivePeaks: e.target.checked })} />
        Live peaks (magenta)
      </label>
      <label className="row">
        <input type="checkbox" checked={overlay.showPredicted}
          onChange={(e) => setOverlay({ ...overlay, showPredicted: e.target.checked })} />
        Predicted reflections (green)
      </label>
      <label className="row">
        <input type="checkbox" checked={overlay.showValues}
          onChange={(e) => setOverlay({ ...overlay, showValues: e.target.checked })} />
        Pixel values (when zoomed)
      </label>
      <label className="row">
        <input type="checkbox" checked={score.show}
          onChange={(e) => setScore({ ...score, show: e.target.checked })} />
        Matched filter {scoreLoading && <span className="muted">… computing</span>}
      </label>

      {score.show && (
        <div className="indent">
          <div className="slider">
            <span>opacity: {score.opacity.toFixed(2)}</span>
            <input type="range" min={0} max={1} step={0.05} value={score.opacity}
              onChange={(e) => setScore({ ...score, opacity: +e.target.value })} />
          </div>
          <div className="slider">
            <span>σ radial: {score.sigmaRadial.toFixed(2)}</span>
            <input type="range" min={0.5} max={5} step={0.1} value={score.sigmaRadial}
              onChange={(e) => setScore({ ...score, sigmaRadial: +e.target.value })} />
          </div>
          <div className="slider">
            <span>σ azimuthal: {score.sigmaAzimuthal.toFixed(2)}</span>
            <input type="range" min={0.3} max={3} step={0.1} value={score.sigmaAzimuthal}
              onChange={(e) => setScore({ ...score, sigmaAzimuthal: +e.target.value })} />
          </div>
        </div>
      )}

      <h3>Contrast (semi-log)</h3>
      <div className="slider">
        <span>p90 (linear→log knee): {contrast.p90.toFixed(0)}</span>
        <input type="range" min={1} max={Math.max(2, hi)} step={1} value={contrast.p90}
          onChange={(e) => setContrast({ ...contrast, p90: +e.target.value })} />
      </div>
      <div className="slider">
        <span>vmax: {contrast.vmax.toFixed(0)}</span>
        <input type="range" min={2} max={Math.max(4, hi * 1.5)} step={1} value={contrast.vmax}
          onChange={(e) => setContrast({ ...contrast, vmax: +e.target.value })} />
      </div>
    </div>
  );
}
