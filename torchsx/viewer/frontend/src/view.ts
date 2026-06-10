// View transform (pan/zoom) + world↔screen + panel affine helpers.
//
// World coordinates are detector pixel units shared by panels, peaks, and
// predictions. A pixel at panel-local (ss, fs) maps to world via the panel
// affine (corner + fs*fs_vec + ss*ss_vec), matching visualize_h5frame.py.

import type { Geometry, Panel } from "./types";

export interface View {
  scale: number;          // screen px per world unit
  cx: number; cy: number; // world coord shown at canvas center
  W: number; H: number;   // canvas size (css px)
}

// Scene coordinates used for display: horizontal = detector world-x, vertical =
// detector world-y. The vertical axis is origin-lower (see worldToScreen).
// Renderer, peaks, and the pixel inverse all go through these helpers, so panels
// and overlays stay aligned.
export function panelWorld(p: Panel, ss: number, fs: number): [number, number] {
  const tx = p.corner_x + fs * p.fs_vec[0] + ss * p.ss_vec[0];
  const ty = p.corner_y + fs * p.fs_vec[1] + ss * p.ss_vec[1];
  return [tx, ty]; // [horizontal, vertical]
}

// World bounding box over all panel corners.
export function worldBounds(geom: Geometry) {
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (const p of geom.panels) {
    for (const [ss, fs] of [[0, 0], [0, p.n_fs], [p.n_ss, p.n_fs], [p.n_ss, 0]]) {
      const [x, y] = panelWorld(p, ss, fs);
      if (x < xmin) xmin = x; if (x > xmax) xmax = x;
      if (y < ymin) ymin = y; if (y > ymax) ymax = y;
    }
  }
  return { xmin, xmax, ymin, ymax };
}

export function fitView(geom: Geometry, W: number, H: number): View {
  const b = worldBounds(geom);
  const wspan = Math.max(b.xmax - b.xmin, 1);
  const hspan = Math.max(b.ymax - b.ymin, 1);
  const scale = 0.95 * Math.min(W / wspan, H / hspan);
  return { scale, cx: (b.xmin + b.xmax) / 2, cy: (b.ymin + b.ymax) / 2, W, H };
}

// scene → canvas pixel (top-left origin; vertical flipped so larger Y is up).
export function worldToScreen(v: View, x: number, y: number): [number, number] {
  return [(x - v.cx) * v.scale + v.W / 2, v.H / 2 - (y - v.cy) * v.scale];
}

// canvas pixel → scene.
export function screenToWorld(v: View, px: number, py: number): [number, number] {
  return [v.cx + (px - v.W / 2) / v.scale, v.cy - (py - v.H / 2) / v.scale];
}

export function zoomAt(v: View, px: number, py: number, factor: number): View {
  const [wx, wy] = screenToWorld(v, px, py);
  const scale = Math.max(1e-4, v.scale * factor);
  // keep (wx,wy) under the cursor after the zoom
  return { ...v, scale, cx: wx - (px - v.W / 2) / scale, cy: wy + (py - v.H / 2) / scale };
}

export function panBy(v: View, dpx: number, dpy: number): View {
  return { ...v, cx: v.cx - dpx / v.scale, cy: v.cy + dpy / v.scale };
}

// Invert the panel affine: scene (X,Y) → (ss, fs) within panel p, or null if
// outside. Scene is (world-x, world-y), so X=true_x, Y=true_y directly.
export function worldToPanelPixel(p: Panel, sceneX: number, sceneY: number): [number, number] | null {
  const [fx, fy] = [p.fs_vec[0], p.fs_vec[1]];
  const [sx, sy] = [p.ss_vec[0], p.ss_vec[1]];
  const det = fx * sy - sx * fy;
  if (Math.abs(det) < 1e-12) return null;
  const dx = sceneX - p.corner_x; // true_x - corner_x
  const dy = sceneY - p.corner_y; // true_y - corner_y
  const fs = (dx * sy - dy * sx) / det;
  const ss = (fx * dy - fy * dx) / det;
  if (ss < 0 || ss >= p.n_ss || fs < 0 || fs >= p.n_fs) return null;
  return [ss, fs];
}

// Locate the panel + integer pixel under a world point (searches all panels).
export function pixelAt(geom: Geometry, x: number, y: number):
  { panel: number; ss: number; fs: number } | null {
  for (let i = 0; i < geom.panels.length; i++) {
    const r = worldToPanelPixel(geom.panels[i], x, y);
    if (r) return { panel: i, ss: Math.floor(r[0]), fs: Math.floor(r[1]) };
  }
  return null;
}
