// 2D-canvas overlay drawn on top of the WebGL layer: found peaks (colored by
// probability) and, when zoomed in far enough, the numeric pixel value in each
// visible pixel. Shares the same View transform as the renderer.

import type { Counts, FoundPeak, Geometry, Mask, Panel, PredReflection, Reflection } from "./types";
import { panelWorld, screenToWorld, View, worldToScreen } from "./view";

// Raw panel-affine inverse (no in-bounds check) for scene coords — bounds the
// value loop. Scene is (world-x, world-y), matching worldToPanelPixel.
function rawInverse(p: Panel, sceneX: number, sceneY: number): [number, number] | null {
  const [fx, fy] = [p.fs_vec[0], p.fs_vec[1]];
  const [sx, sy] = [p.ss_vec[0], p.ss_vec[1]];
  const det = fx * sy - sx * fy;
  if (Math.abs(det) < 1e-12) return null;
  const dx = sceneX - p.corner_x, dy = sceneY - p.corner_y;
  return [(fx * dy - fy * dx) / det, (dx * sy - dy * sx) / det]; // [ss, fs]
}

export interface OverlayOpts {
  showResults: boolean;     // indexed reflections (circles, carry hkl)
  showPeaks: boolean;       // raw peak-finder hits from the stream (red hollow squares)
  showValues: boolean;      // honored only when zoomed past the pixel-label threshold
  showLivePeaks: boolean;   // live re-found peaks (magenta crosses)
  showPredicted: boolean;   // predicted reflections (green circles, carry hkl)
}

// Live layers fed in from a re-run of the pipeline (preview only).
export interface LiveLayers {
  livePeaks: FoundPeak[];
  predicted: PredReflection[];
}

const VALUE_LABEL_MIN_SCALE = 28; // screen px per detector pixel before labels show

export function drawOverlay(
  ctx: CanvasRenderingContext2D,
  view: View,
  geom: Geometry,
  counts: Counts | null,
  reflections: Reflection[],
  peaks: FoundPeak[],
  opts: OverlayOpts,
  live: LiveLayers,
  mask: Mask | null,
) {
  ctx.clearRect(0, 0, view.W, view.H);

  if (opts.showValues && counts && view.scale >= VALUE_LABEL_MIN_SCALE) {
    drawPixelValues(ctx, view, geom, counts, mask);
  }
  if (opts.showPeaks) {
    drawFoundPeaks(ctx, view, geom, peaks);
  }
  if (opts.showLivePeaks) {
    drawLivePeaks(ctx, view, geom, live.livePeaks);
  }
  if (opts.showPredicted) {
    drawPredicted(ctx, view, geom, live.predicted);
  }
  if (opts.showResults) {
    drawReflections(ctx, view, geom, reflections);
  }
}

// Live re-found peaks — magenta crosses, distinct from the stream's red squares
// and the green predicted circles.
function drawLivePeaks(ctx: CanvasRenderingContext2D, view: View, geom: Geometry, peaks: FoundPeak[]) {
  const r = Math.max(3, Math.min(11, view.scale * 2.2));
  ctx.strokeStyle = "#ff4dff";
  ctx.lineWidth = 1.3;
  for (const pk of peaks) {
    const panel = geom.panels[pk.panel];
    if (!panel) continue;
    const [wx, wy] = panelWorld(panel, pk.ss - panel.min_ss, pk.fs);
    const [sx, sy] = worldToScreen(view, wx, wy);
    if (sx < -20 || sx > view.W + 20 || sy < -20 || sy > view.H + 20) continue;
    ctx.beginPath();
    ctx.moveTo(sx - r, sy); ctx.lineTo(sx + r, sy);
    ctx.moveTo(sx, sy - r); ctx.lineTo(sx, sy + r);
    ctx.stroke();
  }
}

// Predicted reflections — green hollow circles (carry hkl + excitation error).
function drawPredicted(ctx: CanvasRenderingContext2D, view: View, geom: Geometry, refl: PredReflection[]) {
  const radius = Math.max(3, Math.min(12, view.scale * 2.6));
  ctx.strokeStyle = "#4dff88";
  ctx.lineWidth = 1.2;
  for (const r of refl) {
    const panel = geom.panels[r.panel];
    if (!panel) continue;
    const [wx, wy] = panelWorld(panel, r.ss - panel.min_ss, r.fs);
    const [sx, sy] = worldToScreen(view, wx, wy);
    if (sx < -20 || sx > view.W + 20 || sy < -20 || sy > view.H + 20) continue;
    ctx.beginPath();
    ctx.arc(sx, sy, radius, 0, Math.PI * 2);
    ctx.stroke();
  }
}

// Raw found peaks (hits) — red hollow squares, distinct from the indexed-reflection
// circles. Drawn under the reflections so an indexed spot shows both marks.
function drawFoundPeaks(ctx: CanvasRenderingContext2D, view: View, geom: Geometry, peaks: FoundPeak[]) {
  const r = Math.max(3, Math.min(10, view.scale * 2.2));
  ctx.strokeStyle = "#ff3b30";
  ctx.lineWidth = 1.2;
  for (const pk of peaks) {
    const panel = geom.panels[pk.panel];
    if (!panel) continue;
    const [wx, wy] = panelWorld(panel, pk.ss - panel.min_ss, pk.fs);
    const [sx, sy] = worldToScreen(view, wx, wy);
    if (sx < -20 || sx > view.W + 20 || sy < -20 || sy > view.H + 20) continue;
    // hollow square centered on the peak (no fill → underlying pixels stay visible)
    ctx.strokeRect(sx - r, sy - r, 2 * r, 2 * r);
  }
}

function probColor(prob: number): string {
  // low prob → orange, high prob → cyan
  const t = Math.max(0, Math.min(1, prob));
  return `rgb(${Math.round(255 * (1 - t))},${Math.round(180 + 40 * t)},${Math.round(80 + 175 * t)})`;
}

function drawReflections(ctx: CanvasRenderingContext2D, view: View, geom: Geometry, refl: Reflection[]) {
  const radius = Math.max(4, Math.min(14, view.scale * 3));
  ctx.lineWidth = 1.5;
  for (const r of refl) {
    const panel = geom.panels[r.panel];
    if (!panel) continue;
    const localSs = r.ss - panel.min_ss; // reflection ss is assembled slow-scan
    const [wx, wy] = panelWorld(panel, localSs, r.fs);
    const [sx, sy] = worldToScreen(view, wx, wy);
    if (sx < -20 || sx > view.W + 20 || sy < -20 || sy > view.H + 20) continue;
    // colored by prob when present, else a fixed indexed-reflection color
    ctx.strokeStyle = r.prob != null ? probColor(r.prob) : "#3fd6ff";
    ctx.beginPath();
    ctx.arc(sx, sy, radius, 0, Math.PI * 2);
    ctx.stroke();
  }
}

function drawPixelValues(ctx: CanvasRenderingContext2D, view: View, geom: Geometry, counts: Counts, mask: Mask | null) {
  const [P, nss, nfs] = counts.shape;
  // Visible world rectangle.
  const [wx0, wy0] = screenToWorld(view, 0, 0);
  const [wx1, wy1] = screenToWorld(view, view.W, view.H);
  const wxmin = Math.min(wx0, wx1), wxmax = Math.max(wx0, wx1);
  const wymin = Math.min(wy0, wy1), wymax = Math.max(wy0, wy1);

  // One detector pixel spans `view.scale` screen px; size the text to 30% of that.
  ctx.font = `${(view.scale * 0.3).toFixed(1)}px monospace`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  const corners: [number, number][] = [[wxmin, wymin], [wxmax, wymin], [wxmax, wymax], [wxmin, wymax]];
  for (let p = 0; p < geom.panels.length && p < P; p++) {
    const panel = geom.panels[p];
    // Map the visible rect's corners into this panel's local (ss,fs) and iterate
    // only that clamped range — bounded by the few pixels actually on screen.
    let ssMin = Infinity, ssMax = -Infinity, fsMin = Infinity, fsMax = -Infinity;
    for (const [cx, cy] of corners) {
      const r = rawInverse(panel, cx, cy);
      if (!r) continue;
      ssMin = Math.min(ssMin, r[0]); ssMax = Math.max(ssMax, r[0]);
      fsMin = Math.min(fsMin, r[1]); fsMax = Math.max(fsMax, r[1]);
    }
    const s0 = Math.max(0, Math.floor(ssMin)), s1 = Math.min(nss - 1, Math.ceil(ssMax));
    const f0 = Math.max(0, Math.floor(fsMin)), f1 = Math.min(nfs - 1, Math.ceil(fsMax));
    if (s0 > s1 || f0 > f1) continue;
    for (let ss = s0; ss <= s1; ss++) {
      for (let fs = f0; fs <= f1; fs++) {
        // +0.5 → pixel center (integer ss/fs is the pixel corner).
        const [wx, wy] = panelWorld(panel, ss + 0.5, fs + 0.5);
        const [px, py] = worldToScreen(view, wx, wy);
        const flat = (p * nss + ss) * nfs + fs;
        const val = counts.data[flat];
        // masked pixels are black in the GL layer → show their value in brackets,
        // in a readable color regardless of count magnitude.
        const masked = mask != null && mask.data[flat] === 0;
        ctx.fillStyle = masked ? "#7fd6ff" : (val > counts.meta.p90 ? "#000" : "#ddd");
        ctx.fillText(masked ? `[${val}]` : String(val), px, py);
      }
    }
  }
}
