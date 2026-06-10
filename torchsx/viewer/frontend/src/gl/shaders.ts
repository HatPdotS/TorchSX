// WebGL2 / GLSL ES 3.00 shaders for the per-panel detector renderer.
//
// Each panel is a quad in world coords with a uint16 data texture (R16UI). The
// fragment shader applies the semi-log brightness transfer (linear 0→p90, log
// p90→vmax) and a viridis colormap.

export const VERT = `#version 300 es
in vec2 a_world;
in vec2 a_uv;
out vec2 v_uv;
uniform vec2 u_center;   // world coord at canvas center
uniform vec2 u_scale;    // [2*scale/W, -2*scale/H]
void main() {
  v_uv = a_uv;
  gl_Position = vec4((a_world - u_center) * u_scale, 0.0, 1.0);
}`;

export const FRAG = `#version 300 es
precision highp float;
precision highp int;
precision highp usampler2D;
in vec2 v_uv;
out vec4 o_color;
uniform usampler2D u_tex;
uniform usampler2D u_mask;   // R8UI good-pixel mask (1=good, 0=masked)
uniform float u_hasMask;     // 1.0 when a mask texture is bound
uniform ivec2 u_texSize;   // (n_fs, n_ss)
uniform float u_p90;
uniform float u_vmax;

// Semi-log transfer → t in [0,1]: linear up to p90 (lower half of the ramp),
// log from p90 to vmax (upper half).
float toT(float v) {
  if (v <= 0.0) return 0.0;
  float p90 = max(u_p90, 1.0);
  float vmax = max(u_vmax, p90 + 1.0);
  if (v <= p90) return 0.5 * (v / p90);
  return clamp(0.5 + 0.5 * (log(v) - log(p90)) / (log(vmax) - log(p90)), 0.0, 1.0);
}

// viridis approximation (polynomial fit, public domain).
vec3 viridis(float t) {
  const vec3 c0 = vec3(0.2777, 0.0054, 0.3340);
  const vec3 c1 = vec3(0.1050, 1.4046, 1.3845);
  const vec3 c2 = vec3(-0.3308, 0.2148, 0.0952);
  const vec3 c3 = vec3(-4.6342, -5.7991, -19.3324);
  const vec3 c4 = vec3(6.2280, 14.1799, 56.6905);
  const vec3 c5 = vec3(4.7763, -13.7451, -65.3530);
  const vec3 c6 = vec3(-5.4354, 4.6453, 26.3124);
  t = clamp(t, 0.0, 1.0);
  return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

void main() {
  ivec2 sz = u_texSize;
  ivec2 coord = ivec2(clamp(v_uv, 0.0, 0.999999) * vec2(sz));
  coord = clamp(coord, ivec2(0), sz - 1);
  // masked pixels render pure black regardless of their (still-present) count
  if (u_hasMask > 0.5 && texelFetch(u_mask, coord, 0).r == 0u) {
    o_color = vec4(0.0, 0.0, 0.0, 1.0);
    return;
  }
  float v = float(texelFetch(u_tex, coord, 0).r);
  o_color = vec4(viridis(toT(v)), 1.0);
}`;

// Matched-filter score overlay: samples an R16F texture, solid red, alpha
// rising with the (gamma-boosted) normalized score so background is transparent
// and signal stands out over the counts layer.
export const SCORE_FRAG = `#version 300 es
precision highp float;
precision highp int;
in vec2 v_uv;
out vec4 o_color;
uniform sampler2D u_tex;   // R16F
uniform ivec2 u_texSize;
uniform float u_smax;
uniform float u_opacity;

void main() {
  ivec2 coord = ivec2(clamp(v_uv, 0.0, 0.999999) * vec2(u_texSize));
  coord = clamp(coord, ivec2(0), u_texSize - 1);
  float v = texelFetch(u_tex, coord, 0).r;
  float t = clamp(v / max(u_smax, 1e-3), 0.0, 1.0);
  t = pow(t, 0.4);                              // boost weak responses
  o_color = vec4(1.0, 0.0, 0.0, u_opacity * t); // solid red, transparent where score ~ 0
}`;
