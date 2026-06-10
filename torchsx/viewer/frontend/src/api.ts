// Typed fetch wrappers for the viewer API. Paths are relative: in dev Vite
// proxies /api to the Python server; in production FastAPI serves both.
import type {
  Counts, CountsMeta, Crystal, FindResult, FoundPeak, Geometry, Manifest, Mask,
  PredictResult, Reflection, Score, Solution,
} from "./types";

async function getJSON<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json() as Promise<T>;
}

async function postJSON<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    let detail = `${r.status}`;
    try { detail = (await r.json()).detail ?? detail; } catch { /* keep status */ }
    throw new Error(`${url}: ${detail}`);
  }
  return r.json() as Promise<T>;
}

// Liveness probe: round-trip the tiny /api/ping and return the latency in ms.
// Throws on any network/HTTP failure so the caller can flag "disconnected".
export async function ping(): Promise<number> {
  const t0 = performance.now();
  const r = await fetch("/api/ping", { cache: "no-store" });
  if (!r.ok) throw new Error(`ping: ${r.status}`);
  await r.arrayBuffer(); // drain the (tiny) body before stopping the clock
  return performance.now() - t0;
}

export const getManifest = () => getJSON<Manifest>("/api/manifest");
export const getGeometry = () => getJSON<Geometry>("/api/geometry");
export const getReflections = (i: number) => getJSON<Reflection[]>(`/api/frames/${i}/reflections`);
export const getPeaks = (i: number) => getJSON<FoundPeak[]>(`/api/frames/${i}/peaks`);
export const getCrystals = (i: number) => getJSON<Crystal[]>(`/api/frames/${i}/crystal`);

// Counts: server sends gzipped uint16 with Content-Encoding: gzip → browser
// decompresses transparently, we get the raw little-endian buffer.
export async function getCounts(i: number): Promise<Counts> {
  const meta = await getJSON<CountsMeta>(`/api/frames/${i}/counts/meta`);
  const r = await fetch(`/api/frames/${i}/counts`);
  if (!r.ok) throw new Error(`counts ${i}: ${r.status}`);
  return { data: new Uint16Array(await r.arrayBuffer()), shape: meta.shape, meta };
}

// Detector good-pixel mask (per-run constant): gzipped uint8, 1=good/0=masked.
// Returns null when the stream carries no mask (server 404). Shape comes from the
// manifest's detector dims since the mask matches the counts layout.
export async function getMask(shape: [number, number, number]): Promise<Mask | null> {
  const r = await fetch("/api/mask");
  if (r.status === 404) return null;
  if (!r.ok) throw new Error(`mask: ${r.status}`);
  return { data: new Uint8Array(await r.arrayBuffer()), shape };
}

// Matched-filter score: gzipped float16; we keep the raw half-float bits for a
// HALF_FLOAT texture. smin/smax ride in response headers.
export async function getScore(
  i: number, sigmaRadial?: number, sigmaAzimuthal?: number,
): Promise<Score> {
  const q = new URLSearchParams();
  if (sigmaRadial != null) q.set("sigma_radial", String(sigmaRadial));
  if (sigmaAzimuthal != null) q.set("sigma_azimuthal", String(sigmaAzimuthal));
  const meta = await getJSON<{ shape: [number, number, number] }>(`/api/frames/${i}/score/meta`);
  const r = await fetch(`/api/frames/${i}/score?${q}`);
  if (!r.ok) throw new Error(`score ${i}: ${r.status}`);
  const data = new Uint16Array(await r.arrayBuffer());
  const smin = parseFloat(r.headers.get("X-Score-Min") ?? "0");
  const smax = parseFloat(r.headers.get("X-Score-Max") ?? "1");
  return { data, shape: meta.shape, smin, smax };
}

// ── live reprocessing (POST; preview only, never writes the stream) ──────────

export const postFindSpots = (
  i: number, body: { top_n: number; sigma_radial?: number; sigma_azimuthal?: number; peak_min_distance?: number },
) => postJSON<FindResult>(`/api/frames/${i}/find-spots`, body);

export const postIndex = (
  i: number, body: { spots?: FoundPeak[]; cell?: number[] | null; min_num_spots?: number },
) => postJSON<Solution>(`/api/frames/${i}/index`, body);

export const postPredict = (
  i: number, body: { basis: number[][]; d_min: number; offset_max: number },
) => postJSON<PredictResult>(`/api/frames/${i}/predict`, body);

// Decode one IEEE half-float (bit pattern) to JS number — for the hover readout.
export function halfToFloat(h: number): number {
  const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}
