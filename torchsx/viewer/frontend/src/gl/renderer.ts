// Per-panel WebGL2 renderer: a counts pass (R16UI + semi-log/viridis) always,
// then an optional matched-filter score pass (R16F + hot colormap, alpha-blended)
// on top. One quad + texture per panel sidesteps the 16384 max-texture-dimension
// limit (the assembled detector is 16448 px tall).

import * as twgl from "twgl.js";
import type { Counts, Geometry, Mask, Score } from "../types";
import type { View } from "../view";
import { panelWorld } from "../view";
import { FRAG, SCORE_FRAG, VERT } from "./shaders";

interface PanelGL {
  buf: twgl.BufferInfo;
  countsTex: WebGLTexture | null;
  scoreTex: WebGLTexture | null;
  maskTex: WebGLTexture | null;
  n_ss: number; n_fs: number;
}

export interface ScoreOpts { show: boolean; opacity: number; smax: number; }

export class Renderer {
  gl: WebGL2RenderingContext;
  countsProg: twgl.ProgramInfo;
  scoreProg: twgl.ProgramInfo;
  panels: PanelGL[] = [];

  constructor(canvas: HTMLCanvasElement) {
    const gl = canvas.getContext("webgl2", { antialias: false, premultipliedAlpha: false });
    if (!gl) throw new Error("WebGL2 not available in this browser");
    this.gl = gl;
    this.countsProg = twgl.createProgramInfo(gl, [VERT, FRAG]);
    this.scoreProg = twgl.createProgramInfo(gl, [VERT, SCORE_FRAG]);
  }

  setGeometry(geom: Geometry) {
    const gl = this.gl;
    this.dispose();
    this.panels = geom.panels.map((p) => {
      const c0 = panelWorld(p, 0, 0), c1 = panelWorld(p, 0, p.n_fs);
      const c2 = panelWorld(p, p.n_ss, p.n_fs), c3 = panelWorld(p, p.n_ss, 0);
      const buf = twgl.createBufferInfoFromArrays(gl, {
        a_world: { numComponents: 2, data: [...c0, ...c1, ...c2, ...c3] },
        a_uv: { numComponents: 2, data: [0, 0, 1, 0, 1, 1, 0, 1] },
        indices: [0, 1, 2, 0, 2, 3],
      });
      return { buf, countsTex: null, scoreTex: null, maskTex: null, n_ss: p.n_ss, n_fs: p.n_fs };
    });
  }

  private _upload(prev: WebGLTexture | null, nfs: number, nss: number,
                  internal: number, format: number, type: number, data: ArrayBufferView) {
    const gl = this.gl;
    if (prev) gl.deleteTexture(prev);
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, internal, nfs, nss, 0, format, type, data);
    return tex;
  }

  setCounts(counts: Counts) {
    const gl = this.gl;
    const [P, nss, nfs] = counts.shape;
    const stride = nss * nfs;
    for (let p = 0; p < this.panels.length && p < P; p++) {
      const sub = counts.data.subarray(p * stride, (p + 1) * stride);
      this.panels[p].countsTex = this._upload(
        this.panels[p].countsTex, nfs, nss, gl.R16UI, gl.RED_INTEGER, gl.UNSIGNED_SHORT, sub);
    }
  }

  // Per-run constant mask (1=good, 0=masked) as a per-panel R8UI texture.
  setMask(mask: Mask) {
    const gl = this.gl;
    const [P, nss, nfs] = mask.shape;
    const stride = nss * nfs;
    for (let p = 0; p < this.panels.length && p < P; p++) {
      const sub = mask.data.subarray(p * stride, (p + 1) * stride);
      this.panels[p].maskTex = this._upload(
        this.panels[p].maskTex, nfs, nss, gl.R8UI, gl.RED_INTEGER, gl.UNSIGNED_BYTE, sub);
    }
  }

  setScore(score: Score | null) {
    const gl = this.gl;
    if (!score) { for (const pg of this.panels) { if (pg.scoreTex) gl.deleteTexture(pg.scoreTex); pg.scoreTex = null; } return; }
    const [P, nss, nfs] = score.shape;
    const stride = nss * nfs;
    for (let p = 0; p < this.panels.length && p < P; p++) {
      const sub = score.data.subarray(p * stride, (p + 1) * stride); // half-float bits
      this.panels[p].scoreTex = this._upload(
        this.panels[p].scoreTex, nfs, nss, gl.R16F, gl.RED, gl.HALF_FLOAT, sub);
    }
  }

  draw(view: View, p90: number, vmax: number, score: ScoreOpts) {
    const gl = this.gl;
    gl.viewport(0, 0, view.W, view.H);
    gl.clearColor(0.05, 0.05, 0.07, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    const uScale = [2 * view.scale / view.W, 2 * view.scale / view.H];
    const uCenter = [view.cx, view.cy];

    // counts pass (opaque)
    gl.disable(gl.BLEND);
    gl.useProgram(this.countsProg.program);
    for (const pg of this.panels) {
      if (!pg.countsTex) continue;
      twgl.setBuffersAndAttributes(gl, this.countsProg, pg.buf);
      twgl.setUniforms(this.countsProg, {
        u_center: uCenter, u_scale: uScale, u_tex: pg.countsTex,
        // bind a real texture for u_mask even when absent (sampled only if u_hasMask>0)
        u_mask: pg.maskTex ?? pg.countsTex, u_hasMask: pg.maskTex ? 1 : 0,
        u_texSize: [pg.n_fs, pg.n_ss], u_p90: p90, u_vmax: vmax,
      });
      twgl.drawBufferInfo(gl, pg.buf);
    }

    // score pass (alpha-blended over counts)
    if (score.show && this.panels.some((p) => p.scoreTex)) {
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.useProgram(this.scoreProg.program);
      for (const pg of this.panels) {
        if (!pg.scoreTex) continue;
        twgl.setBuffersAndAttributes(gl, this.scoreProg, pg.buf);
        twgl.setUniforms(this.scoreProg, {
          u_center: uCenter, u_scale: uScale, u_tex: pg.scoreTex,
          u_texSize: [pg.n_fs, pg.n_ss], u_smax: score.smax, u_opacity: score.opacity,
        });
        twgl.drawBufferInfo(gl, pg.buf);
      }
      gl.disable(gl.BLEND);
    }
  }

  dispose() {
    for (const pg of this.panels) {
      if (pg.countsTex) this.gl.deleteTexture(pg.countsTex);
      if (pg.scoreTex) this.gl.deleteTexture(pg.scoreTex);
      if (pg.maskTex) this.gl.deleteTexture(pg.maskTex);
    }
  }
}
