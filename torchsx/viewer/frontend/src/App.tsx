import { useEffect, useRef, useState } from "react";
import * as api from "./api";
import { ConnectionStatus } from "./components/ConnectionStatus";
import { Controls, ScoreControls } from "./components/Controls";
import { DetectorView, HoverInfo } from "./components/DetectorView";
import { FrameBrowser } from "./components/FrameBrowser";
import { InfoPanel } from "./components/InfoPanel";
import { Reprocess } from "./components/Reprocess";
import type { OverlayOpts } from "./overlay";
import type {
  Counts, Crystal, FindResult, FoundPeak, Geometry, Manifest, Mask, PredictResult,
  Reflection, Score, Solution,
} from "./types";

export function App() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [geom, setGeom] = useState<Geometry | null>(null);
  const [mask, setMask] = useState<Mask | null>(null);
  const [idx, setIdx] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [counts, setCounts] = useState<Counts | null>(null);
  const [reflections, setReflections] = useState<Reflection[]>([]);
  const [peaks, setPeaks] = useState<FoundPeak[]>([]);
  const [crystals, setCrystals] = useState<Crystal[]>([]);
  const [nPeaks, setNPeaks] = useState<number | null>(null);

  const [contrast, setContrast] = useState({ p90: 4, vmax: 1000 });
  const [overlay, setOverlay] = useState<OverlayOpts>({
    showResults: true, showPeaks: true, showValues: true, showLivePeaks: true, showPredicted: true,
  });
  const [score, setScoreCtl] = useState<ScoreControls>({ show: false, opacity: 0.7, sigmaRadial: 2.0, sigmaAzimuthal: 0.7 });
  const [scoreData, setScoreData] = useState<Score | null>(null);
  const [scoreLoading, setScoreLoading] = useState(false);
  const [hover, setHover] = useState<HoverInfo | null>(null);

  // live reprocessing results (preview overlays) — cleared when the frame changes
  const [livePeaks, setLivePeaks] = useState<FindResult | null>(null);
  const [solution, setSolution] = useState<Solution | null>(null);
  const [predicted, setPredicted] = useState<PredictResult | null>(null);

  useEffect(() => {
    Promise.all([api.getManifest(), api.getGeometry()])
      .then(([m, g]) => {
        setManifest(m); setGeom(g);
        setScoreCtl((s) => ({ ...s, sigmaRadial: m.score_default.sigma_radial, sigmaAzimuthal: m.score_default.sigma_azimuthal }));
        if (m.frames.length) setIdx(m.frames[0].idx);
        // mask is a per-run constant (1=good, 0=masked); null when the stream has none
        const d = m.detector;
        api.getMask([d.n_panels, d.n_ss, d.n_fs]).then(setMask).catch(() => setMask(null));
      })
      .catch((e) => setError(String(e)));
  }, []);

  // per-frame cached data
  useEffect(() => {
    if (idx == null) return;
    let cancelled = false;
    setHover(null); setScoreData(null);
    // live reprocessing layers are frame-specific — drop them on frame change
    setLivePeaks(null); setSolution(null); setPredicted(null);
    Promise.all([api.getCounts(idx), api.getReflections(idx), api.getPeaks(idx), api.getCrystals(idx)])
      .then(([c, refl, pks, cry]) => {
        if (cancelled) return;
        setCounts(c); setReflections(refl); setPeaks(pks); setCrystals(cry);
        setNPeaks(manifest?.frames.find((f) => f.idx === idx)?.n_peaks ?? null);
        setContrast({ p90: Math.max(1, c.meta.p90), vmax: Math.max(2, c.meta.vmax) });
      })
      .catch((e) => setError(String(e)));
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [idx]);

  // matched-filter score: fetch when shown (and on frame / sigma change), debounced
  const scoreReq = useRef(0);
  useEffect(() => {
    if (idx == null || !score.show) { setScoreData(null); return; }
    const req = ++scoreReq.current;
    setScoreLoading(true);
    const t = setTimeout(() => {
      api.getScore(idx, score.sigmaRadial, score.sigmaAzimuthal)
        .then((s) => { if (req === scoreReq.current) setScoreData(s); })
        .catch((e) => setError(String(e)))
        .finally(() => { if (req === scoreReq.current) setScoreLoading(false); });
    }, 250);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [idx, score.show, score.sigmaRadial, score.sigmaAzimuthal]);

  if (error) return <div className="fatal">Error: {error}</div>;
  if (!manifest || !geom) return <div className="loading">Loading…</div>;

  const scoreOpts = { show: score.show, opacity: score.opacity, smax: scoreData?.smax ?? 1 };

  return (
    <div className="app">
      <aside className="sidebar">
        <h1>torchsx viewer</h1>
        <ConnectionStatus />
        <FrameBrowser manifest={manifest} idx={idx ?? -1} onSelect={setIdx} />
        <Controls
          meta={counts?.meta ?? null}
          contrast={contrast} setContrast={setContrast}
          overlay={overlay} setOverlay={setOverlay}
          score={score} setScore={setScoreCtl} scoreLoading={scoreLoading}
        />
        <Reprocess
          idx={idx} streamCrystal={crystals[0] ?? null}
          find={livePeaks} setFind={setLivePeaks}
          solution={solution} setSolution={setSolution}
          predicted={predicted} setPredicted={setPredicted}
        />
        <InfoPanel crystals={crystals} nPeaks={nPeaks} hover={hover} />
      </aside>
      <main className="stage">
        <DetectorView
          geom={geom} counts={counts} mask={mask} reflections={reflections} peaks={peaks} score={scoreData}
          live={{ livePeaks: livePeaks?.peaks ?? [], predicted: predicted?.reflections ?? [] }}
          contrast={contrast} overlay={overlay} scoreOpts={scoreOpts} onHover={setHover}
        />
      </main>
    </div>
  );
}
