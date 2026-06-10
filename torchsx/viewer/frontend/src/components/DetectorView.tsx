// Interactive detector canvas: a WebGL layer (counts + optional score) with a
// 2D-canvas overlay (reflections + pixel values) on top, sharing one View.
// View state lives in a ref so pan/zoom never re-renders React.

import { useEffect, useRef } from "react";
import { Renderer, ScoreOpts } from "../gl/renderer";
import { drawOverlay, LiveLayers, OverlayOpts } from "../overlay";
import { halfToFloat } from "../api";
import type { Counts, FoundPeak, Geometry, Mask, PredReflection, Reflection, Score } from "../types";
import { fitView, panBy, panelWorld, pixelAt, screenToWorld, View, worldToScreen, zoomAt } from "../view";

export interface HoverInfo {
  panel: number; ss: number; fs: number; value: number; score: number | null;
  masked?: boolean;
  pred?: { hkl: [number, number, number]; excitation_error: number } | null;
}

// Nearest predicted reflection within HOVER_PX screen px of (sx,sy), for the
// hover readout (hkl + excitation error). O(n) per move — fine for hover.
const HOVER_PX = 10;
function nearestPredicted(geom: Geometry, view: View, pred: PredReflection[], sx: number, sy: number) {
  let best: PredReflection | null = null, bestD = HOVER_PX * HOVER_PX;
  for (const r of pred) {
    const panel = geom.panels[r.panel];
    if (!panel) continue;
    const [wx, wy] = panelWorld(panel, r.ss - panel.min_ss, r.fs);
    const [px, py] = worldToScreen(view, wx, wy);
    const d = (px - sx) * (px - sx) + (py - sy) * (py - sy);
    if (d < bestD) { bestD = d; best = r; }
  }
  return best ? { hkl: best.hkl, excitation_error: best.excitation_error } : null;
}

interface Props {
  geom: Geometry;
  counts: Counts | null;
  mask: Mask | null;
  reflections: Reflection[];
  peaks: FoundPeak[];
  live: LiveLayers;
  score: Score | null;
  contrast: { p90: number; vmax: number };
  overlay: OverlayOpts;
  scoreOpts: ScoreOpts;
  onHover: (h: HoverInfo | null) => void;
}

export function DetectorView(props: Props) {
  const { geom } = props;
  const glCanvas = useRef<HTMLCanvasElement>(null);
  const ovCanvas = useRef<HTMLCanvasElement>(null);
  const wrap = useRef<HTMLDivElement>(null);
  const renderer = useRef<Renderer | null>(null);
  const view = useRef<View | null>(null);
  const raf = useRef(0);

  const live = useRef(props);
  live.current = props;

  const requestDraw = () => {
    if (raf.current) return;
    raf.current = requestAnimationFrame(() => {
      raf.current = 0;
      const r = renderer.current, v = view.current;
      if (!r || !v) return;
      const { contrast, overlay, scoreOpts, geom, counts, reflections, peaks, live: layers, mask } = live.current;
      r.draw(v, contrast.p90, contrast.vmax, scoreOpts);
      const ov = ovCanvas.current?.getContext("2d");
      if (ov) drawOverlay(ov, v, geom, counts, reflections, peaks, overlay, layers, mask);
    });
  };

  const resize = () => {
    const el = wrap.current, gc = glCanvas.current, oc = ovCanvas.current;
    if (!el || !gc || !oc) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const W = el.clientWidth, H = el.clientHeight;
    for (const c of [gc, oc]) { c.width = W * dpr; c.height = H * dpr; c.style.width = `${W}px`; c.style.height = `${H}px`; }
    if (view.current) { view.current.W = gc.width; view.current.H = gc.height; }
    requestDraw();
  };

  useEffect(() => {
    if (!glCanvas.current) return;
    try { renderer.current = new Renderer(glCanvas.current); }
    catch (e) { alert((e as Error).message); return; }
    renderer.current.setGeometry(geom);
    resize();
    view.current = fitView(geom, glCanvas.current.width, glCanvas.current.height);
    requestDraw();
    const ro = new ResizeObserver(resize);
    if (wrap.current) ro.observe(wrap.current);
    return () => { ro.disconnect(); renderer.current?.dispose(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [geom]);

  useEffect(() => {
    if (renderer.current && props.counts) { renderer.current.setCounts(props.counts); requestDraw(); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.counts]);

  useEffect(() => {
    if (renderer.current && props.mask) { renderer.current.setMask(props.mask); requestDraw(); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.mask]);

  useEffect(() => {
    if (renderer.current) { renderer.current.setScore(props.score); requestDraw(); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.score]);

  useEffect(requestDraw, [props.contrast, props.overlay, props.scoreOpts, props.reflections, props.peaks, props.live]);

  // ── interaction ────────────────────────────────────────────────────────
  const drag = useRef<{ x: number; y: number } | null>(null);
  const dpr = () => Math.min(window.devicePixelRatio || 1, 2);

  const onWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    if (!view.current) return;
    const rect = glCanvas.current!.getBoundingClientRect();
    view.current = zoomAt(view.current, (e.clientX - rect.left) * dpr(), (e.clientY - rect.top) * dpr(),
      e.deltaY < 0 ? 1.15 : 1 / 1.15);
    requestDraw();
  };

  const onDown = (e: React.PointerEvent) => {
    (e.target as Element).setPointerCapture(e.pointerId);
    drag.current = { x: e.clientX, y: e.clientY };
  };

  const onMove = (e: React.PointerEvent) => {
    if (!view.current) return;
    if (drag.current) {
      const ddx = (e.clientX - drag.current.x) * dpr(), ddy = (e.clientY - drag.current.y) * dpr();
      drag.current = { x: e.clientX, y: e.clientY };
      view.current = panBy(view.current, ddx, ddy);
      requestDraw();
      return;
    }
    const rect = glCanvas.current!.getBoundingClientRect();
    const sx = (e.clientX - rect.left) * dpr(), sy = (e.clientY - rect.top) * dpr();
    const [wx, wy] = screenToWorld(view.current, sx, sy);
    const hit = pixelAt(live.current.geom, wx, wy);
    const c = live.current.counts, s = live.current.score;
    const pred = live.current.overlay.showPredicted
      ? nearestPredicted(live.current.geom, view.current, live.current.live.predicted, sx, sy)
      : null;
    if (hit && c) {
      const [, nss, nfs] = c.shape;
      const flat = (hit.panel * nss + hit.ss) * nfs + hit.fs;
      const score = s ? halfToFloat(s.data[flat]) : null;
      const m = live.current.mask;
      const masked = m != null && m.data[flat] === 0;
      props.onHover({ ...hit, value: c.data[flat], score, masked, pred });
    } else props.onHover(null);
  };

  const onUp = (e: React.PointerEvent) => { drag.current = null; (e.target as Element).releasePointerCapture?.(e.pointerId); };

  return (
    <div ref={wrap} className="detector-view">
      <canvas ref={glCanvas} className="layer" />
      <canvas ref={ovCanvas} className="layer overlay"
        onWheel={onWheel} onPointerDown={onDown} onPointerMove={onMove}
        onPointerUp={onUp} onPointerLeave={() => props.onHover(null)} />
    </div>
  );
}
