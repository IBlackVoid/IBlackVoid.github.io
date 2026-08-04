
/* void.js — the terminal.
 *
 * The whole page is one display. A voxel world is raymarched into an offscreen
 * buffer sized to the character grid, then a second pass reads exactly one
 * texel per cell and stamps a glyph on it. That split is the entire trick: the
 * expensive pass (a real 3D march) runs at ~480x270, and the only thing that
 * ever touches full resolution is a handful of texture reads. It is also how a
 * terminal actually works, which is the point.
 *
 * No libraries. No build. WebGL1 so it runs on anything.
 */
(() => {
"use strict";

/* ---------------------------------------------------------------- config -- */

const STATIONS = 6;          // must match the number of <section class=station>
const CELL_CSS = 9;          // glyph cell, in CSS pixels
const DPR_CAP  = 2;
const MAX_CANVAS_PIXELS = 3000000;
const MAX_STEPS = 112;
const MIN_STEPS = 44;        // below this the far field visibly truncates
const MIN_QUALITY = 0.45;

// Cell coordinates are world units, so bounds are read directly by the marcher.
// Tight bounds are the main perf lever: a ray that misses the box never steps.
const BAND     = 0.17;       // half-width of a station transition, in stations

const WORLDS = [
  { min: [-48, 0, -48], max: [ 48, 24,  48] },  // 0 relief
  { min: [-48, 0,   2], max: [ 48, 14,  30] },  // 1 histogram
  { min: [-48, 0, -30], max: [ 48, 14,  30] },  // 2 split
  { min: [-52, 0, -18], max: [ 52, 32,  18] },  // 3 commit topology
  { min: [-46, 0, -14], max: [ 46, 28,  14] },  // 4 project machines
  { min: [-64, 0, -24], max: [ 64, 38,  24] },  // 5 proof / tea / mark
];

/* ---------------------------------------------------------------- shaders -- */

const VERT = `
attribute vec2 aPos;
void main(){ gl_Position = vec4(aPos, 0.0, 1.0); }`;

/* Pass 1 — the world. Amanatides & Woo grid traversal: exact, branch-free per
 * step, and it hands back the face normal for free. Sphere tracing would blur
 * the silhouettes we specifically want hard.
 *
 * There is one world, not six. Every station stands on the same ruled ground
 * plane at y=0, and a station boundary is a front sweeping across that ground
 * rewriting what is on it — the old station ahead of the front, the new one
 * behind. Nothing ever cuts, and nothing ever fades to an empty screen.
 */
const SCENE = `
precision highp float;

uniform vec2      uRes, uShift;
uniform vec3      uCamPos, uCamTgt, uUMin, uUMax;
uniform float     uTime, uWorldA, uWorldB, uMorph, uPhase, uCut, uFocus,
                  uInflate, uSplit, uZoom, uMaxSteps, uAfter, uRitual,
                  uObserve;
uniform sampler2D uRelief, uMark;

const vec3 VOIDC = vec3(0.047, 0.047, 0.047);
const vec3 CYAN  = vec3(0.251, 0.627, 0.659);
const vec3 AMBER = vec3(0.816, 0.408, 0.125);
const vec3 SHINE = vec3(0.973, 0.925, 0.847);
const vec3 GREEN = vec3(0.314, 0.784, 0.439);
const vec3 STEEL = vec3(0.470, 0.600, 0.640);
const vec3 BLOOD = vec3(0.580, 0.025, 0.020);
const vec3 TEAL  = vec3(0.100, 0.680, 0.700);
const vec3 TEA   = vec3(0.820, 0.390, 0.090);

float h11(float p){ return fract(sin(p * 127.1) * 43758.5453123); }
float h21(vec2  p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123); }
float h31(vec3  p){ return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453123); }

float lumOf(vec3 c){ return dot(c, vec3(0.299, 0.587, 0.114)); }

/* Each station's footprint. Kept in the shader rather than passed in because
 * during a transition two of them are live at once and the marcher only gets
 * their union. Mirrored by WORLDS in the script above. */
void wbounds(float w, out vec3 lo, out vec3 hi){
  if      (w < 0.5){ lo = vec3(-48.0, 0.0,-48.0); hi = vec3( 48.0, 24.0, 48.0); }
  else if (w < 1.5){ lo = vec3(-48.0, 0.0,  2.0); hi = vec3( 48.0, 14.0, 30.0); }
  else if (w < 2.5){ lo = vec3(-48.0, 0.0,-30.0); hi = vec3( 48.0, 14.0, 30.0); }
  else if (w < 3.5){ lo = vec3(-52.0, 0.0,-18.0); hi = vec3( 52.0, 32.0, 18.0); }
  else if (w < 4.5){ lo = vec3(-46.0, 0.0,-14.0); hi = vec3( 46.0, 28.0, 14.0); }
  else             { lo = vec3(-64.0, 0.0,-24.0); hi = vec3( 64.0, 38.0, 24.0); }
}

/* Exact inverse CDF for the six measured self-compile buckets:
 * 78,438 / 15,948 / 8,333 / 9,259 / 1,881 / 1,603 requests.
 * Rows are a deterministic reconstruction at those observed proportions. */
float reqLen(float u){
  if (u < 0.67934039) return   2.0 + floor(u / 0.67934039 * 30.0);
  if (u < 0.81746375) return  33.0 + floor((u - 0.67934039) / 0.13812336 * 15.0);
  if (u < 0.88963469) return  49.0 + floor((u - 0.81746375) / 0.07217094 * 15.0);
  if (u < 0.96982557) return  65.0 + floor((u - 0.88963469) / 0.08019088 * 63.0);
  if (u < 0.98611664) return 129.0 + floor((u - 0.96982557) / 0.01629107 * 127.0);
  return                         257.0 + floor((u - 0.98611664) / 0.01388336 * 127.0);
}

/* 462 of 463 measured stack-eligible string-to-rune conversions stayed at or
 * below 32 elements. This lane is in rune elements, not bytes. */
float runeLen(float u){
  if (u < 0.99784017) return 1.0 + floor(u / 0.99784017 * 31.0);
  return 65.0 + floor((u - 0.99784017) / 0.00215983 * 63.0);
}

/* The ground. Ruled every eight cells, shared by every station, and the reason
 * the page reads as one place: the floor never changes, only what stands on it. */
float ground(vec3 c){
  if (c.x < -84.0 || c.x > 114.0 || c.z < -64.0 || c.z > 64.0) return 0.0;
  if (mod(c.x, 8.0) < 1.0 || mod(c.z, 8.0) < 1.0) return 13.0;
  return 0.0;
}

/* Material id for one cell of one station. 0 is empty. Returning an id rather
 * than a colour keeps the inner loop to a single texture read. */
float worldAt(float w, vec3 c){
  vec3 lo, hi;
  wbounds(w, lo, hi);
  if (any(lessThan(c, lo)) || any(greaterThan(c, hi))) return 0.0;
  if (c.y < 1.0) return ground(c);

  float m = 0.0;

  if (w < 0.5){
    // 00 — the relief. His own Pixel-Ripper frame laid on the ground as
    // terrain, luminance as height, so the render stops being a picture and
    // becomes a landscape the camera descends over.
    vec2 uv = vec2((c.x - lo.x) / (hi.x - lo.x), (c.z - lo.z) / (hi.z - lo.z));
    float l = lumOf(texture2D(uRelief, uv).rgb);
    float h = 1.0 + floor((0.03 + 0.97 * pow(l, 1.12)) * (hi.y - 2.0) * uInflate);
    if (c.y <= h) m = 1.0;

  } else if (w < 1.5){
    // 01 — a deterministic reconstruction sampled from the six observed
    // bucket counts. The bright wall is the old 32-byte constant; every
    // oxide cell past it represents the measured spill distribution.
    float row = c.z + floor(uPhase);
    float b   = c.x - lo.x;                       // byte index along the row
    if (abs(b - 32.0) < 0.5 && c.y < 9.0) return 4.0;
    // Every other row left empty. Packed solid the field reads as one sheet
    // instead of a stack of separate measurements.
    if (mod(row, 2.0) >= 1.0) return 0.0;
    float L   = reqLen(h11(row * 1.37 + 3.1));
    float top = 2.0 + ((b < 32.0) ? 0.0 : floor((b - 31.0) * 0.42));
    if (b < L && c.y < top) m = b < 32.0 ? 2.0 : 3.0;

  } else if (w < 2.5){
    // The upper lane keeps byte units. The lower lane keeps rune-element units:
    // its wall at 32 therefore means 32 runes, or 128 bytes of stack.
    float wall = mix(32.0, 64.0, uSplit);
    float row  = c.z + floor(uPhase);
    float b    = c.x - lo.x;
    if (c.z > 1.0){
      if (abs(b - wall) < 0.5 && c.y < 2.0 + 7.0 * max(0.05, uSplit)) return 4.0;
      if (mod(row, 2.0) >= 1.0) return 0.0;
      float L   = reqLen(h11(row * 1.37 + 3.1));
      float top = 2.0 + ((b < wall) ? 0.0 : floor((b - wall + 1.0) * 0.42));
      if (b < L && c.y < top) m = b < wall ? 2.0 : 3.0;
    } else if (c.z < -1.0 && h11(row * 1.19 + 7.0) < uSplit){
      if (abs(b - 32.0) < 0.5 && c.y < 2.0 + 7.0 * uSplit) return 4.0;
      if (mod(row, 2.0) >= 1.0) return 0.0;
      float L = runeLen(h11(row * 2.11 + 9.7));
      if (b < L && c.y < 1.0 + 2.0 * uSplit) m = 5.0;
    }

  } else if (w < 3.5){
    // A steel trunk carries the merged Go commit. Three review branches leave
    // it without closing: they remain shells until they land.
    if (abs(c.z) < 1.2 && c.x > -48.0 && c.x < 48.0 &&
        c.y >= 2.0 && c.y < 4.0) m = 14.0;

    if (abs(c.x + 36.0) < 5.0 && abs(c.z) < 5.0 && c.y < 28.0){
      m = 6.0;
    } else {
      float i = floor((c.x + 22.0) / 24.0);
      float cx = -10.0 + i * 24.0;
      if (i >= 0.0 && i < 3.0){
        float branchY = 4.0 + c.z * 0.66;
        if (abs(c.x - cx) < 1.0 && c.z > 0.0 && c.z < 11.0 &&
            abs(c.y - branchY) < 1.0) m = 7.0;
        vec3 q = c - vec3(cx, 13.0 + i * 2.0, 12.0);
        vec3 aq = abs(q);
        bool inside = max(aq.x, max(aq.y, aq.z)) < 5.0;
        bool shell = max(aq.x, max(aq.y, aq.z)) > 3.7;
        if (inside && shell) m = 7.0;
      }
    }

  } else if (w < 4.5){
    float i = floor((c.x + 46.0) / 23.0);
    vec3 q = c - vec3(-34.5 + i * 23.0, 14.0, 0.0);

    if (i < 0.5){
      // VoiDex: an audio waveform rises into timed subtitle layers.
      float wave = sin((q.x + uTime * 4.0) * 0.72) * 2.6;
      if (abs(q.y + 8.0) < 1.0 && abs(q.z - wave) < 1.0 && abs(q.x) < 10.0) m = 9.0;
      if (q.y > -3.0 && q.y < 9.0 && mod(q.y + 3.0, 4.0) < 1.0 &&
          abs(q.x) < 8.5 && abs(q.z) < 6.0) m = 9.0;

    } else if (i < 1.5){
      // Pixel-Ripper: source luminance becomes a field of glyph-bearing cells.
      vec2 uv = vec2(q.x / 18.0 + 0.5, q.y / 20.0 + 0.5);
      float l = lumOf(texture2D(uRelief, uv).rgb);
      if (abs(q.z) < 1.0 && abs(q.x) < 9.0 && abs(q.y) < 10.0 &&
          h21(floor(q.xy) + 4.0) < l * 1.18) m = 10.0;
      float sweep = mod(uTime * 4.0, 22.0) - 11.0;
      if (abs(q.x - sweep) < 0.8 && abs(q.y) < 10.0 && abs(q.z) < 2.0) m = 4.0;

    } else if (i < 2.5){
      // Aethelred: three rectangular namespace boundaries, nested and explicit.
      vec3 aq = abs(q);
      float d0 = max(aq.x / 10.0, max(aq.y / 10.0, aq.z / 10.0));
      float d1 = max(aq.x /  7.0, max(aq.y /  7.0, aq.z /  7.0));
      float d2 = max(aq.x /  4.0, max(aq.y /  4.0, aq.z /  4.0));
      if (abs(d0 - 1.0) < 0.10 || abs(d1 - 1.0) < 0.14 || abs(d2 - 1.0) < 0.24) m = 11.0;

    } else {
      // Helios: replicas around a primary, with a write pulse in flight.
      float a  = atan(q.z, q.x);
      float nd = floor((a + 3.14159) / 0.7854);
      float na = -3.14159 + nd * 0.7854 + 0.3927;
      vec3 node = vec3(cos(na) * 8.0, 0.0, sin(na) * 8.0);
      if (length(q - node) < 2.6) m = 12.0;
      float ta = uTime * 1.1;
      if (length(q - vec3(cos(ta) * 8.0, 0.0, sin(ta) * 8.0)) < 1.8) m = 4.0;
      if (abs(q.x) < 1.0 && abs(q.z) < 1.0 && abs(q.y) < 7.0) m = 12.0;
    }

  } else {
    // 05 — public proof, a discovered tea ritual, and the final signature.
    float ritual = smoothstep(0.02, 0.98, uRitual);
    if (ritual > 0.001 && uAfter < 0.54){
      // Jane's low turquoise cup is authored in one local coordinate system.
      // Keeping every dimension in q is essential: mixing q.xz with world y
      // was what turned the previous vessel into a swollen cylinder.
      float lift = smoothstep(0.025, 0.15, uAfter);
      vec3 q = c - vec3(-14.0 - 2.0 * lift, 1.0 + 28.0 * lift, 0.0);
      float y = q.y;
      float r2 = dot(q.xz, q.xz);
      float cupMat = 0.0;

      float t = clamp((y - 3.1) / 13.4, 0.0, 1.0);
      float outer = mix(6.2, 8.6, smoothstep(0.0, 1.0, t))
                  + 0.55 * sin(3.14159 * t);
      float inner = outer - 1.05;
      if (y >= 3.1 && y < 16.5 &&
          r2 < outer * outer && r2 > inner * inner) cupMat = 16.0;
      if (y >= 2.8 && y < 4.2 && r2 < 42.25) cupMat = 16.0;

      // Two shallow ridges catch light without turning the bowl into a stack
      // of rings. They echo the saucer and the period cup's ceramic tooling.
      if (cupMat > 15.5 &&
          (abs(y - 7.0) < 0.34 || abs(y - 12.8) < 0.28)) cupMat = 19.0;

      if (y >= 15.9 && y < 17.4 && r2 < 86.49 && r2 > 57.00)
        cupMat = 20.0;
      if (y >= 16.15 && y < 16.75 && r2 < 58.52) cupMat = 17.0;

      // The ring handle and its two bridges are separate tests so the hole
      // remains open at character resolution without detaching from the bowl.
      vec2 hp = vec2((q.x - 10.1) / 4.75, (y - 10.1) / 5.25);
      float hd = dot(hp, hp);
      if (q.x > 7.2 && abs(q.z) < 1.25 && hd > 0.56 && hd < 1.30)
        cupMat = 16.0;
      if (q.x >= 6.8 && q.x < 10.2 && abs(q.z) < 1.35 &&
          ((y > 6.4 && y < 8.1) || (y > 12.5 && y < 14.2)))
        cupMat = 16.0;

      float bodyBreak = smoothstep(0.105, 0.16, uAfter);
      if (cupMat > 0.0 &&
          (bodyBreak < 0.001 || h31(c * 1.31 + 19.0) > bodyBreak))
        m = cupMat;

      // The saucer stays behind when the cup lifts. A circular object in world
      // space becomes the correct ellipse through perspective; drawing an
      // ellipse into the geometry was another source of the old icon look.
      vec3 sq = c - vec3(-14.0, 1.0, 0.0);
      float sr = length(sq.xz);
      float dishY = 0.8 + 1.35 * pow(clamp(sr / 15.9, 0.0, 1.0), 2.0);
      float saucerMat = 0.0;
      if (sr < 15.9 && sq.y >= dishY && sq.y < dishY + 1.15)
        saucerMat = 16.0;
      if (saucerMat > 0.0 &&
          (abs(sr - 10.8) < 0.48 || abs(sr - 13.6) < 0.42))
        saucerMat = 19.0;
      float saucerBreak = smoothstep(0.22, 0.34, uAfter);
      if (saucerMat > 0.0 &&
          (saucerBreak < 0.001 || h31(c * 0.77 + 31.0) > saucerBreak))
        m = saucerMat;

      // Two unequal steam threads. Camera parallax supplies the restrained
      // pointer bend, and the accepted-letter scalar advances their glint.
      float steamLife = 1.0 - smoothstep(0.02, 0.075, uAfter);
      if (steamLife > 0.001 && y >= 18.0 && y < 35.0){
        float bend = clamp((uCamPos.x - 18.0) / 5.0, -1.0, 1.0);
        float rise = smoothstep(18.0, 35.0, y);
        float d0 = length(vec2(
          q.x + 3.1 - sin(y * 0.31 + uTime * 0.31) * 1.15 - bend * rise * 2.2,
          q.z - cos(y * 0.23 + 0.8) * 0.48));
        float d1 = length(vec2(
          q.x - 2.4 - sin(y * 0.27 + uTime * 0.24 + 2.1) * 0.92 - bend * rise * 1.7,
          q.z + 0.7 - cos(y * 0.19 + 1.6) * 0.42));
        if (min(d0, d1) < 0.86 && mod(y + uObserve * 2.0, 3.0) < 2.0)
          m = 18.0;
      }
    }

    // The tell is the ring: the saucer rotates toward the camera, expands,
    // and loses its ceramic colour before the irregular painted gesture takes
    // over. One geometry transition connects the calm ritual to the threat.
    float ringIn = smoothstep(0.20, 0.28, uAfter);
    float ringOut = 1.0 - smoothstep(0.48, 0.56, uAfter);
    float ringLife = ritual * ringIn * ringOut;
    if (ringLife > 0.001){
      float turn = smoothstep(0.22, 0.48, uAfter);
      vec3 rq = c - vec3(mix(-14.0, 0.0, turn), 1.0, 0.0);
      float angle = turn * 1.5707963;
      float ca = cos(angle), sa = sin(angle);
      float normal = ca * rq.y + sa * rq.z;
      float radial = -sa * rq.y + ca * rq.z;
      float rr = length(vec2(rq.x, radial));
      float radius = mix(13.6, 30.0, smoothstep(0.22, 0.52, uAfter));
      float inkWidth = mix(0.18, mix(0.65, 1.35, turn), ringIn) * ringOut;
      if (abs(rr - radius) < max(0.10, inkWidth) && abs(normal) < 1.15)
        m = 16.0;
    }

    vec2 uv = vec2((c.x - lo.x) / (hi.x - lo.x), 1.0 - (c.y - 2.0) / 34.0);
    if (uv.y > 0.0 && uv.y < 1.0 && abs(c.z) < 1.5){
      vec4 mark = texture2D(uMark, uv);
      float proofKeep = h31(c * 0.41 + 7.0);
      if (mark.r > 0.5 && proofKeep > ritual) m = 8.0;

      float write = smoothstep(0.28, 0.42, uAfter);
      float nameErase = smoothstep(0.46, 0.58, uAfter);
      float sign  = smoothstep(0.62, 0.96, uAfter);
      float nameFront = uv.x + uv.y * 0.06;
      if (mark.g > 0.5 && nameFront < write * 1.08 &&
          uv.y + uv.x * 0.04 > nameErase * 1.04) m = 8.0;

      // Blue intensity is the construction order shared with the plaster SVG:
      // one pressure-loaded ring, the two smears, the heavy mouth, then gravity.
      float strokeGate = mix(1.01, 0.46, sign);
      if (mark.b > strokeGate) m = 15.0;
    }
  }

  return m;
}

/* Where the sweep front is for a given cell. Negative is still the old
 * station, positive has been rewritten. The per-cell jitter makes the edge
 * ragged at character scale instead of a ruler-straight wipe. */
float front(vec3 c){
  float u = (c.x - uUMin.x) / max(1.0, uUMax.x - uUMin.x);
  return (uMorph * 1.36 - 0.18) - u + (h31(c * 0.83) - 0.5) * 0.20;
}

float solid(vec3 c){
  float w = uWorldA;
  if (abs(uWorldA - uWorldB) > 0.1) w = (front(c) > 0.0) ? uWorldB : uWorldA;
  float m = worldAt(w, c);
  if (m > 0.0 && uCut > 0.0 && h31(c * 0.37 + 11.0) < uCut) return 0.0;
  return m;
}

/* Glyph ink covers maybe a tenth of a cell, so a value that looks correct here
 * arrives on screen an order of magnitude darker. Everything below is graded
 * for what survives the character pass, not for what looks right in the buffer. */
vec3 matColor(float m, vec3 c){
  if (m < 1.5){
    vec3 lo, hi; wbounds(0.0, lo, hi);
    vec2 uv = vec2((c.x - lo.x) / (hi.x - lo.x), (c.z - lo.z) / (hi.z - lo.z));
    vec3 t = texture2D(uRelief, uv).rgb;
    float sourceLuma = lumOf(t);
    vec3 sourceColour = vec3(sourceLuma) + (t - vec3(sourceLuma)) * 1.82;
    float lift = pow(sourceLuma, 0.78);
    return clamp(sourceColour, 0.0, 1.35) * (1.42 + 0.86 * lift)
         + SHINE * 0.025;
  }
  if (m < 2.5)  return CYAN  * (1.05 + 0.45 * h31(c * 0.7));
  if (m < 3.5)  return AMBER * (1.16 + 0.55 * h31(c * 0.7));
  if (m < 4.5)  return SHINE * 1.45;
  if (m < 5.5)  return STEEL * (1.00 + 0.38 * h31(c * 0.7));
  if (m < 6.5)  return GREEN * (1.28 + 0.28 * h31(c * 0.5));
  if (m < 7.5)  return AMBER * 1.08;
  if (m < 8.5)  return SHINE * (1.20 + 0.32 * h31(c * 0.4));
  if (m < 9.5)  return CYAN * ((abs(uFocus) < 0.5) ? 2.10 : 1.04);
  if (m < 10.5) return vec3(0.95, 0.64, 0.36) * ((abs(uFocus - 1.0) < 0.5) ? 2.10 : 1.02);
  if (m < 11.5) return STEEL * ((abs(uFocus - 2.0) < 0.5) ? 2.20 : 1.12);
  if (m < 12.5) return vec3(0.68, 0.56, 0.86) * ((abs(uFocus - 3.0) < 0.5) ? 2.10 : 1.04);
  if (m < 13.5) return STEEL * 0.25;
  if (m < 14.5) return STEEL * 0.76;
  if (m < 15.5) return BLOOD * 1.64;
  if (m < 16.5) return mix(
    TEAL * (1.26 + 0.16 * h31(c * 0.47) + 0.10 * uObserve),
    BLOOD * 1.58,
    smoothstep(0.25, 0.39, uAfter));
  if (m < 17.5) return mix(TEA * 1.32, BLOOD * 0.92,
                           smoothstep(0.10, 0.34, uAfter)) *
                           (1.0 + 0.08 * uObserve);
  if (m < 18.5) return SHINE * (0.58 + 0.22 * h31(c * 0.31));
  if (m < 19.5) return mix(
    vec3(0.035, 0.320, 0.360) * 1.36,
    BLOOD * 1.36,
    smoothstep(0.25, 0.39, uAfter));
  return vec3(0.480, 0.940, 0.890) * 1.30;
}

void main(){
  // uShift slides the whole frustum sideways so the subject can sit clear of
  // the copy without the camera having to lie about where it is looking.
  vec2 p = (gl_FragCoord.xy * 2.0 - uRes) / uRes.y + uShift;

  vec3 fw = normalize(uCamTgt - uCamPos);
  vec3 rt = normalize(cross(fw, vec3(0.0, 1.0, 0.0)));
  vec3 up = cross(rt, fw);
  vec3 rd = normalize(p.x * rt + p.y * up + uZoom * fw);
  vec3 ro = uCamPos;

  // Axis-aligned rays make 1/rd explode; nudging them off the axis is cheaper
  // than branching inside the traversal loop.
  vec3 srd = sign(rd) + vec3(equal(rd, vec3(0.0)));
  vec3 ard = max(abs(rd), vec3(1e-5));
  rd  = srd * ard;
  vec3 ri = 1.0 / rd;

  // Clip to the union of the live stations before stepping. A ray that misses
  // costs nothing, which is what keeps the budget for the ones that hit.
  vec3 t0 = (uUMin - ro) * ri, t1 = (uUMax + 1.0 - ro) * ri;
  vec3 ta = min(t0, t1), tb = max(t0, t1);
  float tEnter = max(max(ta.x, ta.y), max(ta.z, 0.0));
  float tExit  = min(min(tb.x, tb.y), tb.z);

  vec3  col   = vec3(0.0);
  float depth = 1.0;
  float mat   = 0.0;
  vec3  hitC  = vec3(0.0);
  vec3  n     = vec3(0.0);
  float dist  = 0.0;

  if (tExit > tEnter){
    vec3 org  = ro + rd * (tEnter + 1e-3);
    vec3 pos  = floor(org);
    vec3 rs   = sign(rd);
    vec3 dis  = (pos - org + 0.5 + rs * 0.5) * ri;
    vec3 mask = vec3(0.0, 0.0, 1.0);

    for (int i = 0; i < ${MAX_STEPS}; i++){
      // GLSL ES 1.0 requires a constant loop bound, so the ceiling is baked in
      // and the real budget is a uniform break. That lets the frame-time
      // governor trade march distance at runtime — far cheaper than dropping
      // resolution, because this loop is the whole per-fragment cost.
      if (float(i) >= uMaxSteps) break;
      float m = solid(pos);
      if (m > 0.0){ mat = m; hitC = pos; break; }
      mask = step(dis.xyz, dis.yzx) * step(dis.xyz, dis.zxy);
      dis += mask * ri * rs;
      pos += mask * rs;
      if (any(lessThan(pos, uUMin)) || any(greaterThan(pos, uUMax))) break;
    }

    if (mat > 0.0){
      n = -mask * rs;
      vec3 mini = (hitC - ro + 0.5 - 0.5 * rs) * ri;
      dist = max(mini.x, max(mini.y, mini.z));

      vec3 base = matColor(mat, hitC);
      float dif = max(dot(n, normalize(vec3(0.40, 0.80, 0.46))), 0.0);
      float sky = max(n.y, 0.0);
      float sourceLight = 0.62 + 0.44 * dif + 0.18 * sky;
      float worldLight = 0.34 + 0.62 * dif + 0.26 * sky;
      col  = base * ((mat < 1.5) ? sourceLight : worldLight);
      // A neutral silhouette lift survives the glyph quantisation without
      // imposing a synthetic colour fringe on the source render.
      col += base * 0.18 * pow(max(0.0, -dot(n, rd)), 2.0);

      // The write head: cells being rewritten right now flare, so a station

      // change reads as something happening rather than something replaced.
      if (uMorph > 0.002 && uMorph < 0.998)
        col += SHINE * exp(-abs(front(hitC)) * 22.0) * 0.85;

      col  = mix(col, VOIDC * 1.6, 1.0 - exp(-dist * 0.0062));
      depth = clamp(dist / 200.0, 0.0, 1.0);
    }
  }

  if (mat < 0.5){
    float horizon = smoothstep(-0.35, 0.75, p.y);
    col = VOIDC * (0.62 + 0.32 * horizon);
  }

  gl_FragColor = vec4(col, depth);
}`;

/* Pass 2 — the glyph terminal. One texel of the world per character cell. */
const ASCII = `
precision highp float;

uniform sampler2D uScene, uAtlas;
uniform vec2      uRes, uGrid, uMouse;
uniform float     uCell, uCount, uTime, uBoot, uLens, uLensR,
                  uMotion, uWarpK, uReveal;

const vec3 AQUA = vec3(0.455, 0.847, 0.808);

float h21(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123); }

void main(){
  vec2 px = gl_FragCoord.xy;
  vec2 toMouse = px - uMouse;
  float d = length(toMouse) / max(uRes.x, uRes.y);
  float pull = uWarpK * uMotion * exp(-d * 4.6);
  vec2 warp = normalize(toMouse + 1e-4) * pull *
              (4.0 + 2.2 * sin(uTime * 1.15 - d * 16.0));

  vec2 g = (px + warp) / uCell;
  vec2 cid = floor(g);
  vec2 inCell = fract(g);
  vec2 sceneUV = (cid + 0.5) / uGrid;
  vec4 scene = texture2D(uScene, sceneUV);
  float lum = dot(scene.rgb, vec3(0.299, 0.587, 0.114));

  float density = clamp(pow(lum * 1.35, 0.62) * (1.0 - 0.34 * scene.a), 0.0, 1.0);
  float glyph = floor(density * (uCount - 1.0) + 0.5);

  float randomCell = h21(cid);
  float resolved = smoothstep(0.0, 0.42,
    uBoot * 1.85 - length(sceneUV - 0.5) * 0.62 - randomCell * 0.44);
  float randomGlyph = floor(fract(randomCell * 17.3 + floor(uTime * 15.0) * 0.371) * uCount);
  glyph = clamp(mix(randomGlyph, glyph, resolved), 0.0, uCount - 1.0);

  float ink = texture2D(uAtlas, vec2((glyph + inCell.x) / uCount, inCell.y)).a;
  vec3 col = scene.rgb * (ink * 1.52 + pow(ink, 0.45) * 0.52);
  col += ink * lum * vec3(0.02, 0.07, 0.07);

  float radial = length(px / uRes - 0.5) * 1.42;
  float vignette = 1.0 - smoothstep(0.52, 1.62, radial);
  col *= 0.72 + 0.28 * vignette;

  // The first two seconds expose the shaded source field before it resolves
  // into calibrated glyphs. This is a 3D material reveal, not a flat image
  // crossfade: the source has already been raised into the voxel terrain.
  if (uReveal > 0.001){
    vec3 raw = texture2D(uScene, (px + warp) / (uGrid * uCell)).rgb;
    float sourceReveal = uReveal * (0.72 + 0.28 * smoothstep(0.04, 0.66, lum));
    col = mix(col, raw * 1.18, sourceReveal);
  }

  if (uLens > 0.001){
    float radius = length(px - uMouse);
    float lensMask = (1.0 - smoothstep(uLensR * 0.80, uLensR, radius)) * uLens;
    if (lensMask > 0.001){
      vec3 raw = texture2D(uScene, (px + warp) / (uGrid * uCell)).rgb;
      col = mix(col, raw * 1.16, lensMask);
    }
    float ring = (1.0 - smoothstep(0.0, 2.5, abs(radius - uLensR * 0.90))) * uLens;
    col += AQUA * ring * 0.62;
  }

  gl_FragColor = vec4(max(col, 0.0), 1.0);
}`;

/* ------------------------------------------------------------------ boot -- */

const root   = document.documentElement;
const canvas = document.getElementById("void");
const still  = matchMedia("(prefers-reduced-motion: reduce)");

/* ------------------------------------------------------- observation gate -- */

/* The public document ends before this section. A new gesture must continue
 * downward after the browser has already reached that end; arriving there by
 * inertia is not enough. Arming only expands ordinary document flow. We never
 * cancel wheel, touch, or keyboard defaults, so the ritual cannot become a
 * scroll trap. This is a discovery gate, not access control: public JavaScript
 * can always be inspected. The private deployment is the actual privacy layer. */
const afterElement = document.getElementById("afterimage");
const ritualRoom = document.querySelector(".ritual-room");
const ritualCup = document.getElementById("ritual-cup");
const ritualStatus = document.getElementById("ritual-status");
const ritualSlots = Array.from(document.querySelectorAll(".ritual-slots li"));
const ritualDialog = document.getElementById("ritual-dialog");
const ritualForm = document.getElementById("ritual-form");
const ritualAnswer = document.getElementById("ritual-answer");
const ritualDialogStatus = document.getElementById("ritual-dialog-status");
const ritualCancel = document.getElementById("ritual-cancel");
const afterName = document.querySelector(".afterimage-name");
const afterWord = document.getElementById("afterimage-word");
const afterSignature = document.getElementById("afterimage-signature");
const afterReturn = document.querySelector(".afterimage-return");
const ANSWER = [0x4a, 0x41, 0x4e, 0x45];

/* One source of truth feeds both the plaster SVG and the blue channel of the
 * voxel mask. The silhouettes carry pressure; their centerlines only reveal
 * them. That separation is why the mark can write like a hand gesture without
 * collapsing back into a constant-width vector icon. */
const SIGNATURE = Object.freeze({
  width: 500,
  height: 570,
  parts: Object.freeze([
    {
      id: "ring",
      phase: "circle",
      tone: 255,
      revealWidth: 66,
      d: "M148 8 C205 2 279 28 336 67 C391 105 418 166 407 233 C397 299 359 351 304 377 C239 406 166 382 117 333 C72 288 78 216 102 160 C125 108 165 81 202 65 C213 60 221 64 223 70 C226 77 221 83 211 87 L204 90 C169 104 140 134 123 177 C104 225 108 274 143 310 C182 350 241 367 295 344 C345 322 378 279 385 226 C391 173 369 125 321 91 C275 59 210 35 162 36 C149 36 138 29 138 21 C138 14 142 10 148 8 Z",
      reveal: "M150 21 C208 17 283 43 329 77 C379 113 402 167 396 228 C389 285 355 334 300 359 C242 385 176 365 129 323 C88 285 91 221 113 167 C133 121 170 94 208 77",
    },
    {
      id: "top-trail",
      phase: "accent",
      tone: 245,
      revealWidth: 13,
      d: "M203 0 C240 6 286 24 323 49 C329 53 331 57 328 60 C325 63 319 60 313 56 C278 34 238 19 201 8 C195 6 197 1 203 0 Z",
      reveal: "M202 4 C244 11 286 28 322 54",
    },
    {
      id: "ring-drain",
      phase: "ring-drip",
      tone: 228,
      revealWidth: 10,
      d: "M303 370 C310 374 311 387 310 404 L311 466 C312 510 310 548 308 568 C307 572 303 571 302 566 C300 545 304 510 303 469 L302 407 C301 389 298 377 303 370 Z",
      reveal: "M305 373 C307 419 305 489 306 567",
    },
    {
      id: "right-eye",
      phase: "eye-right",
      tone: 214,
      revealWidth: 43,
      d: "M266 173 L299 157 C308 159 316 165 324 172 L337 186 C339 192 335 199 331 201 C325 198 319 190 310 187 C300 183 292 184 284 190 C277 195 273 204 268 207 C264 202 263 181 266 173 Z",
      reveal: "M268 187 L300 167 C312 169 322 177 333 190",
    },
    {
      id: "left-eye",
      phase: "eye-left",
      tone: 205,
      revealWidth: 40,
      d: "M159 188 L173 177 C181 174 190 176 198 181 L204 184 L212 179 C217 178 221 181 222 185 C222 190 219 196 215 200 C209 204 203 200 198 201 C190 202 185 211 175 212 C167 211 160 204 157 197 C156 193 157 190 159 188 Z",
      reveal: "M161 194 L176 181 C185 180 193 187 201 192 L217 187",
    },
    {
      id: "right-eye-drip",
      phase: "eye-right-drip",
      tone: 184,
      revealWidth: 9,
      d: "M313 184 C318 188 320 197 319 209 L320 258 C320 270 318 279 315 279 C311 278 312 265 312 255 L311 212 C310 200 308 191 313 184 Z",
      reveal: "M314 188 C316 217 315 249 316 278",
    },
    {
      id: "left-eye-drip",
      phase: "eye-left-drip",
      tone: 174,
      revealWidth: 10,
      d: "M160 198 C166 202 168 211 167 224 L166 286 C166 307 165 323 162 325 C158 322 159 305 159 287 L158 229 C157 216 155 205 160 198 Z",
      reveal: "M161 201 C163 239 160 286 162 324",
    },
    {
      id: "mouth",
      phase: "smile",
      tone: 160,
      revealWidth: 47,
      d: "M180 273 C190 273 200 286 214 292 C239 303 276 300 301 282 L312 271 C317 267 323 275 321 281 C318 289 307 294 300 300 C275 318 239 323 210 312 C193 306 181 298 177 287 C174 280 175 275 180 273 Z",
      reveal: "M181 282 C215 310 269 316 316 279",
    },
    {
      id: "mouth-drip-left",
      phase: "mouth-drips",
      tone: 148,
      revealWidth: 9,
      d: "M192 297 C198 301 199 310 198 322 L199 344 C199 352 196 357 193 352 C190 347 192 336 191 326 C190 314 188 303 192 297 Z",
      reveal: "M194 301 C195 320 193 338 195 353",
    },
    {
      id: "mouth-drip-mid",
      phase: "mouth-drips-mid",
      tone: 140,
      revealWidth: 8,
      d: "M202 304 C207 307 207 316 206 325 L207 340 C206 347 203 348 201 343 C199 337 201 328 200 321 C199 313 198 307 202 304 Z",
      reveal: "M203 307 C204 320 202 335 203 345",
    },
    {
      id: "mouth-drip-right",
      phase: "mouth-drips-late",
      tone: 132,
      revealWidth: 9,
      d: "M226 312 C232 314 233 324 232 335 L233 361 C233 369 230 372 227 367 C224 360 226 348 225 338 C224 326 222 317 226 312 Z",
      reveal: "M228 315 C229 333 227 351 229 368",
    },
    {
      id: "left-ring-drip",
      phase: "last-drip",
      tone: 120,
      revealWidth: 9,
      d: "M187 359 C192 362 193 370 192 380 L193 394 C193 401 190 404 187 400 C184 395 186 386 185 378 C184 369 183 363 187 359 Z",
      reveal: "M189 362 C189 375 188 389 189 400",
    },
  ]),
});

const afterState = {
  raw: 0,
  seen: 0,
  target: 0,
  start: 0,
  end: 1,
  revealStart: 0,
  unlocked: false,
  onchange: null,
};

const ritualState = {
  phase: "sealed",
  typed: [],
  typedTimer: 0,
  wheelPressure: 0,
  wheelSamples: 0,
  wheelStarted: 0,
  wheelActive: false,
  lastWheel: 0,
  keyPressure: 0,
  keyDeadline: 0,
  touchReady: false,
  touchStartY: 0,
  touchPressure: 0,
};

let ritualTarget = 0;
let observeTarget = 0;

const range = (a, b, x) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

function isInteractiveEvent(event){
  const path = typeof event.composedPath === "function"
    ? event.composedPath()
    : [event.target];
  return path.some(node => node instanceof Element && node.matches(
    "a,button,input,textarea,select,summary,[role='textbox']," +
    "[contenteditable]:not([contenteditable='false'])"
  ));
}

function isAtDocumentEnd(){
  const scroller = document.scrollingElement || document.documentElement;
  const remaining = scroller.scrollHeight - window.innerHeight - window.scrollY;
  return remaining <= Math.max(3, window.innerHeight * 0.003);
}

function announce(message){
  if (!ritualStatus) return;
  ritualStatus.textContent = "";
  requestAnimationFrame(() => { ritualStatus.textContent = message; });
}

function resetPressure(){
  ritualState.wheelPressure = 0;
  ritualState.wheelSamples = 0;
  ritualState.wheelStarted = 0;
  ritualState.wheelActive = false;
  ritualState.keyPressure = 0;
  ritualState.keyDeadline = 0;
  ritualState.touchReady = false;
  ritualState.touchPressure = 0;
}

function updateSlots(length, error = false){
  observeTarget = Math.min(1, Math.max(0, length / ANSWER.length));
  if (ritualCup) ritualCup.style.setProperty("--cup-observe", observeTarget.toFixed(2));
  const slotList = ritualSlots[0]?.parentElement;
  if (slotList) slotList.setAttribute(
    "aria-label", `Four-letter observation, ${length} of ${ANSWER.length} noted`
  );
  ritualSlots.forEach((slot, index) => {
    slot.classList.toggle("is-filled", index < length);
    slot.classList.toggle("is-error", error);
  });
  if (error) setTimeout(() => {
    ritualSlots.forEach(slot => slot.classList.remove("is-error"));
  }, 320);
  if (afterState.onchange) afterState.onchange();
}

function clearTyped(){
  ritualState.typed.length = 0;
  if (ritualState.typedTimer) clearTimeout(ritualState.typedTimer);
  ritualState.typedTimer = 0;
  updateSlots(0);
}

function setRitualPhase(phase){
  ritualState.phase = phase;
  root.dataset.ritual = phase;
  ritualTarget = phase === "sealed" ? 0 : 1;
  if (afterState.onchange) afterState.onchange();
}

function armRitual(){
  if (!afterElement || ritualState.phase !== "sealed") return;
  setRitualPhase("armed");
  afterElement.inert = false;
  afterElement.removeAttribute("inert");
  afterElement.setAttribute("aria-hidden", "false");
  if (ritualRoom) ritualRoom.setAttribute("aria-hidden", "false");
  if (afterName) afterName.setAttribute("aria-hidden", "true");
  if (afterSignature) afterSignature.setAttribute("aria-hidden", "true");
  if (afterReturn) afterReturn.setAttribute("aria-hidden", "true");
  if (ritualCup) ritualCup.tabIndex = 0;
  resetPressure();
  announce("A quiet room opened below.");
  requestAnimationFrame(() => {
    measureAfter();
    readAfter();
  });
}

function buildSignature(){
  if (!afterSignature || afterSignature.firstElementChild) return;
  const NS = "http://www.w3.org/2000/svg";
  const make = (name, attrs = {}) => {
    const node = document.createElementNS(NS, name);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    return node;
  };
  const svg = make("svg", {
    viewBox: `0 0 ${SIGNATURE.width} ${SIGNATURE.height}`,
    role: "img",
    "aria-labelledby": "signature-title signature-desc",
  });
  const title = make("title", { id: "signature-title" });
  title.textContent = "A wet crimson face painted on old plaster";
  const desc = make("desc", { id: "signature-desc" });
  desc.textContent = "One heavy clockwise ring, two uneven eyes, a broad smile, and paint still descending under gravity.";
  svg.append(title, desc);

  const defs = make("defs");
  const filter = make("filter", { id: "ink-waver", x: "-3%", y: "-3%", width: "106%", height: "106%" });
  filter.append(
    make("feTurbulence", { type: "fractalNoise", baseFrequency: "0.018 0.055", numOctaves: "2", seed: "11", result: "noise" }),
    make("feDisplacementMap", { in: "SourceGraphic", in2: "noise", scale: "1.6", xChannelSelector: "R", yChannelSelector: "G" })
  );
  defs.append(filter);
  for (const part of SIGNATURE.parts){
    const shape = make("path", { id: `signature-${part.id}`, d: part.d });
    defs.append(shape);

    const mask = make("mask", {
      id: `signature-reveal-${part.id}`,
      maskUnits: "userSpaceOnUse",
      x: "0",
      y: "0",
      width: String(SIGNATURE.width),
      height: String(SIGNATURE.height),
    });
    mask.append(
      make("rect", {
        x: "0",
        y: "0",
        width: String(SIGNATURE.width),
        height: String(SIGNATURE.height),
        fill: "black",
      }),
      make("path", {
        class: `signature-reveal signature-${part.phase}`,
        d: part.reveal,
        pathLength: "1",
        fill: "none",
        stroke: "white",
        "stroke-width": String(part.revealWidth),
        "stroke-linecap": "round",
        "stroke-linejoin": "round",
      })
    );
    defs.append(mask);
  }
  svg.append(defs);

  const appendUses = (parent, className) => {
    for (const part of SIGNATURE.parts){
      parent.append(make("use", {
        class: `${className} signature-${part.phase}`,
        href: `#signature-${part.id}`,
        mask: `url(#signature-reveal-${part.id})`,
      }));
    }
  };

  const bleed = make("g", { class: "signature-bleed", "aria-hidden": "true" });
  appendUses(bleed, "signature-paint");
  const ink = make("g", {
    class: "signature-ink",
    filter: "url(#ink-waver)",
    "aria-hidden": "true",
  });
  appendUses(ink, "signature-paint");
  svg.append(bleed, ink);
  afterSignature.append(svg);
}

function unlockRitual(){
  if (!afterElement || ritualState.phase !== "armed") return;
  clearTyped();
  buildSignature();
  setRitualPhase("unlocked");
  afterState.unlocked = true;
  afterState.seen = 0;
  afterState.revealStart = window.scrollY;
  root.dataset.after = "tea";
  if (afterWord) afterWord.textContent = String.fromCharCode(...ANSWER);

  // A closed modal restores focus to the control that opened it. Move focus
  // to the newly revealed section before hiding that control's room, or the
  // browser can leave the active element inside an aria-hidden subtree.
  afterElement.focus({ preventScroll: true });
  if (ritualRoom) ritualRoom.setAttribute("aria-hidden", "true");
  if (ritualCup) ritualCup.tabIndex = -1;
  if (afterName) afterName.setAttribute("aria-hidden", "false");
  if (afterSignature) afterSignature.setAttribute("aria-hidden", "true");
  announce(`Observation accepted. ${String.fromCharCode(...ANSWER)}. Keep going.`);
  requestAnimationFrame(() => {
    measureAfter();
    commitAfter(0);
  });
}

function sealRitual(moveToEnd = true){
  if (!afterElement || ritualState.phase === "sealed") return;
  clearTyped();
  afterState.unlocked = false;
  afterState.revealStart = 0;
  afterState.seen = 0;
  commitAfter(0);
  setRitualPhase("sealed");
  const active = document.activeElement;
  if (active instanceof HTMLElement && afterElement.contains(active)) active.blur();
  afterElement.inert = true;
  afterElement.setAttribute("inert", "");
  afterElement.setAttribute("aria-hidden", "true");
  if (ritualRoom) ritualRoom.setAttribute("aria-hidden", "true");
  if (ritualCup) ritualCup.tabIndex = -1;
  if (afterName) afterName.setAttribute("aria-hidden", "true");
  if (afterWord) afterWord.textContent = "";
  if (afterSignature){
    afterSignature.setAttribute("aria-hidden", "true");
    afterSignature.replaceChildren();
  }
  if (afterReturn){
    afterReturn.tabIndex = -1;
    afterReturn.setAttribute("aria-hidden", "true");
  }
  if (ritualDialog && ritualDialog.open) ritualDialog.close();
  resetPressure();
  if (moveToEnd){
    const footer = document.querySelector(".site-footer");
    if (footer) footer.scrollIntoView({ block: "end" });
  }
}

function acceptTypedCode(code){
  if (ritualState.phase !== "armed") return;
  let index = ritualState.typed.length;
  if (code === ANSWER[index]){
    ritualState.typed.push(code);
  } else {
    updateSlots(0, ritualState.typed.length > 0);
    ritualState.typed = code === ANSWER[0] ? [code] : [];
  }
  updateSlots(ritualState.typed.length);
  if (ritualState.typedTimer) clearTimeout(ritualState.typedTimer);
  if (ritualState.typed.length === ANSWER.length){
    unlockRitual();
    return;
  }
  ritualState.typedTimer = setTimeout(clearTyped, 3200);
}

function commitAfter(raw){
  afterState.raw = Math.min(1, Math.max(0, raw));
  // Wet paint does not climb back into the brush when the visitor scrolls up.
  // Once a part of the signature has been observed it remains until the room
  // is explicitly resealed.
  if (afterState.unlocked) afterState.seen = Math.max(afterState.seen, afterState.raw);
  else afterState.seen = 0;
  const reduced = still.matches || root.classList.contains("no-motion");
  const p = reduced
    ? (afterState.seen < 0.24 ? 0 : afterState.seen < 0.68 ? 0.54 : 1)
    : afterState.seen;
  if (Math.abs(p - afterState.target) < 0.00005 &&
      !(p === 0 && root.dataset.after !== "sealed")) return;
  afterState.target = p;

  const tea = 1 - range(0.08, 0.28, p);
  const name = range(0.28, 0.40, p) * (1 - range(0.46, 0.58, p));
  const wall = range(0.52, 0.64, p);
  const face = range(0.62, 0.96, p);
  root.style.setProperty("--after", p.toFixed(4));
  root.style.setProperty("--after-chrome",
    (0.22 * (1 - range(0.54, 0.84, p))).toFixed(4));
  root.style.setProperty("--after-tea", tea.toFixed(4));
  root.style.setProperty("--after-name", name.toFixed(4));
  root.style.setProperty("--after-wall", wall.toFixed(4));
  root.style.setProperty("--after-face", face.toFixed(4));
  root.style.setProperty("--after-circle", range(0.56, 0.74, p).toFixed(4));
  root.style.setProperty("--after-accent", range(0.61, 0.74, p).toFixed(4));
  root.style.setProperty("--after-ring-drip", range(0.68, 1.00, p).toFixed(4));
  root.style.setProperty("--after-eye-right", range(0.76, 0.84, p).toFixed(4));
  root.style.setProperty("--after-eye-right-drip", range(0.81, 0.92, p).toFixed(4));
  root.style.setProperty("--after-eye-left", range(0.80, 0.88, p).toFixed(4));
  root.style.setProperty("--after-eye-left-drip", range(0.85, 0.97, p).toFixed(4));
  root.style.setProperty("--after-smile", range(0.86, 0.94, p).toFixed(4));
  root.style.setProperty("--after-mouth-drips", range(0.92, 0.97, p).toFixed(4));
  root.style.setProperty("--after-mouth-drips-mid", range(0.94, 0.985, p).toFixed(4));
  root.style.setProperty("--after-mouth-drips-late", range(0.96, 0.997, p).toFixed(4));
  root.style.setProperty("--after-last-drip", range(0.97, 1.00, p).toFixed(4));

  const phase = !afterState.unlocked ? "sealed"
    : p < 0.27 ? "tea"
    : p < 0.58 ? "name"
    : p < 0.60 ? "wall"
    : p < 0.97 ? "signature"
    : "found";
  if (root.dataset.after !== phase) root.dataset.after = phase;
  const wallVisible = phase === "wall" || phase === "signature" || phase === "found";
  if (afterName) afterName.setAttribute("aria-hidden", wallVisible ? "true" : "false");
  if (afterSignature) afterSignature.setAttribute("aria-hidden", wallVisible ? "false" : "true");
  if (afterReturn){
    const found = phase === "found";
    afterReturn.tabIndex = found ? 0 : -1;
    afterReturn.setAttribute("aria-hidden", found ? "false" : "true");
  }
  if (afterState.onchange) afterState.onchange();
}

function measureAfter(){
  if (!afterElement || !afterState.unlocked) return;
  const top = afterElement.getBoundingClientRect().top + window.scrollY;
  afterState.start = Math.max(top, afterState.revealStart);
  afterState.end = Math.max(afterState.start + 1,
    top + afterElement.offsetHeight - window.innerHeight);
}

function readAfter(){
  if (!afterElement || !afterState.unlocked){
    commitAfter(0);
    return;
  }
  commitAfter((window.scrollY - afterState.start) /
              (afterState.end - afterState.start));
}

root.dataset.after = "sealed";
root.dataset.ritual = "sealed";

if (afterElement){
  afterElement.inert = true;

  addEventListener("wheel", event => {
    const now = performance.now();
    const previous = ritualState.lastWheel;
    ritualState.lastWheel = now;
    if (ritualState.phase !== "sealed") return;
    if (event.deltaY <= 0 || !isAtDocumentEnd()){
      resetPressure();
      ritualState.lastWheel = now;
      return;
    }

    // Inertial events keep arriving without a quiet gap. They update
    // lastWheel but cannot begin the deliberate, second gesture.
    if (!ritualState.wheelActive){
      if (previous && now - previous < 180) return;
      ritualState.wheelActive = true;
      ritualState.wheelStarted = now;
      ritualState.wheelPressure = 0;
      ritualState.wheelSamples = 0;
    } else if (now - previous > 520){
      ritualState.wheelStarted = now;
      ritualState.wheelPressure = 0;
      ritualState.wheelSamples = 0;
    }

    const scale = event.deltaMode === WheelEvent.DOM_DELTA_LINE ? 20
      : event.deltaMode === WheelEvent.DOM_DELTA_PAGE ? window.innerHeight
      : 1;
    ritualState.wheelPressure += Math.min(180, event.deltaY * scale);
    ritualState.wheelSamples++;
    if (ritualState.wheelPressure >= 760 && ritualState.wheelSamples >= 4 &&
        now - ritualState.wheelStarted >= 70) armRitual();
  }, { passive: true });

  addEventListener("touchstart", event => {
    if (ritualState.phase !== "sealed" || event.touches.length !== 1 ||
        !isAtDocumentEnd()) return;
    ritualState.touchReady = true;
    ritualState.touchStartY = event.touches[0].clientY;
    ritualState.touchPressure = 0;
  }, { passive: true });

  addEventListener("touchmove", event => {
    if (!ritualState.touchReady || event.touches.length !== 1) return;
    ritualState.touchPressure = Math.max(ritualState.touchPressure,
      ritualState.touchStartY - event.touches[0].clientY);
    if (ritualState.touchPressure >= 150) armRitual();
  }, { passive: true });

  addEventListener("touchend", () => {
    ritualState.touchReady = false;
    ritualState.touchPressure = 0;
  }, { passive: true });

  addEventListener("touchcancel", () => {
    ritualState.touchReady = false;
    ritualState.touchPressure = 0;
  }, { passive: true });

  addEventListener("scroll", () => {
    if (ritualState.phase === "sealed" && !isAtDocumentEnd()) resetPressure();
    readAfter();
  }, { passive: true });
  addEventListener("resize", () => { measureAfter(); readAfter(); }, { passive: true });
  addEventListener("load", () => { measureAfter(); readAfter(); });
  document.querySelectorAll("details").forEach(details => {
    details.addEventListener("toggle", () => { measureAfter(); readAfter(); });
  });
}

addEventListener("keydown", event => {
  if (event.defaultPrevented || event.repeat || event.isComposing ||
      event.keyCode === 229 || event.metaKey || event.ctrlKey || event.altKey ||
      isInteractiveEvent(event)) return;

  if (ritualState.phase === "sealed"){
    const down = event.key === "ArrowDown" || event.key === "PageDown" ||
      event.key === " " || event.key === "Spacebar" || event.key === "End";
    if (!down || !isAtDocumentEnd()) return;
    const now = performance.now();
    if (now > ritualState.keyDeadline) ritualState.keyPressure = 0;
    ritualState.keyPressure++;
    ritualState.keyDeadline = now + 4200;
    if (ritualState.keyPressure >= 4) armRitual();
    return;
  }

  if (ritualState.phase === "armed"){
    if (event.key === "Escape"){
      sealRitual();
      return;
    }
    if (event.key.length !== 1 || !/^[a-z]$/i.test(event.key)) return;
    acceptTypedCode(event.key.toUpperCase().charCodeAt(0));
  }
});

function resetCupPointer(){
  if (!ritualCup) return;
  ritualCup.style.setProperty("--cup-yaw", "0deg");
  ritualCup.style.setProperty("--cup-pitch", "0deg");
  ritualCup.style.setProperty("--steam-bias", "0px");
}

if (ritualCup){
  ritualCup.addEventListener("pointermove", event => {
    if (ritualState.phase !== "armed" || still.matches ||
        root.classList.contains("no-motion")) return;
    const rect = ritualCup.getBoundingClientRect();
    const x = Math.min(1, Math.max(-1,
      ((event.clientX - rect.left) / Math.max(1, rect.width) - 0.5) * 2));
    const y = Math.min(1, Math.max(-1,
      ((event.clientY - rect.top) / Math.max(1, rect.height) - 0.5) * 2));
    ritualCup.style.setProperty("--cup-yaw", `${(x * 2.5).toFixed(2)}deg`);
    ritualCup.style.setProperty("--cup-pitch", `${(-y * 1.8).toFixed(2)}deg`);
    ritualCup.style.setProperty("--steam-bias", `${(x * 4.5).toFixed(2)}px`);
  }, { passive: true });
  ritualCup.addEventListener("pointerleave", resetCupPointer, { passive: true });
  ritualCup.addEventListener("blur", resetCupPointer);
}

if (ritualCup) ritualCup.addEventListener("click", () => {
  if (ritualState.phase !== "armed" || !ritualDialog || !ritualAnswer) return;
  ritualAnswer.value = "";
  if (ritualDialogStatus) ritualDialogStatus.textContent = "";
  ritualDialog.showModal();
  requestAnimationFrame(() => ritualAnswer.focus());
});

if (ritualCancel) ritualCancel.addEventListener("click", () => ritualDialog.close());
if (afterReturn) afterReturn.addEventListener("click", () => sealRitual(false));
if (ritualForm) ritualForm.addEventListener("submit", event => {
  event.preventDefault();
  const value = ritualAnswer.value.trim().toUpperCase();
  const expected = String.fromCharCode(...ANSWER);
  if (value === expected){
    ritualDialog.close();
    ritualState.typed = ANSWER.slice();
    updateSlots(ANSWER.length);
    unlockRitual();
  } else {
    if (ritualDialogStatus) ritualDialogStatus.textContent = "Look again.";
    ritualAnswer.select();
  }
});

const abandonTyped = () => {
  if (ritualState.phase === "armed") clearTyped();
};
addEventListener("blur", abandonTyped);
addEventListener("pagehide", () => {
  if (ritualState.phase === "sealed") abandonTyped();
  else sealRitual(false);
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden) abandonTyped();
});

let gl = null;
try {
  gl = canvas.getContext("webgl", { antialias: false, alpha: false, depth: false,
                                    powerPreference: "high-performance" }) ||
       canvas.getContext("experimental-webgl", { antialias: false, alpha: false });
} catch (e) { gl = null; }

if (!gl) { root.classList.add("no-gl"); return; }
if (new URLSearchParams(location.search).has("debug")) root.classList.add("debug");

let softwareRenderer = false;
try {
  const rendererInfo = gl.getExtension("WEBGL_debug_renderer_info");
  const renderer = rendererInfo
    ? gl.getParameter(rendererInfo.UNMASKED_RENDERER_WEBGL)
    : "";
  softwareRenderer = /swiftshader|llvmpipe|software/i.test(renderer || "");
} catch (error) {
  softwareRenderer = false;
}

function compile(type, src){
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)){
    console.error(gl.getShaderInfoLog(s), src);
    throw new Error("shader");
  }
  return s;
}
function program(vs, fs){
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
  gl.bindAttribLocation(p, 0, "aPos");
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)){
    console.error(gl.getProgramInfoLog(p));
    throw new Error("link");
  }
  return p;
}

let progScene, progAscii;
try {
  progScene = program(VERT, SCENE);
  progAscii = program(VERT, ASCII);
} catch (e) {
  root.classList.remove("gl-ready");
  root.classList.add("no-gl");
  return;
}

const uni = (p, names) => {
  const o = {};
  for (const n of names) o[n] = gl.getUniformLocation(p, "u" + n[0].toUpperCase() + n.slice(1));
  return o;
};
const uS = uni(progScene, ["res","shift","camPos","camTgt","uMin","uMax","time",
                           "worldA","worldB","morph","phase","cut","focus",
                           "inflate","split","zoom","maxSteps","after","ritual",
                           "observe","relief","mark"]);
const uA = uni(progAscii, ["scene","atlas","res","grid","mouse","cell","count",
                           "time","boot","lens","lensR","motion","warpK","reveal"]);

const quad = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, quad);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 3,-1, -1,3]), gl.STATIC_DRAW);
gl.enableVertexAttribArray(0);
gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

/* ------------------------------------------------------------- the atlas -- */

/* The ramp is calibrated at load, not hard-coded: each candidate glyph is drawn
 * once, its ink coverage measured, and the set resampled to be even in
 * coverage. A ramp tuned against one font goes visibly wrong on another, and
 * this page ships no font. */
function buildAtlas(){
  const CANDIDATES = " .'`^\",:;!i~+_-?][}{)(|/\\tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";
  const probe = document.createElement("canvas");
  probe.width = probe.height = 32;
  const pc = probe.getContext("2d", { willReadFrequently: true });
  const font = w => `${w} 26px ui-monospace,"DejaVu Sans Mono",Menlo,Consolas,monospace`;

  const measured = [];
  for (const ch of CANDIDATES){
    pc.clearRect(0, 0, 32, 32);
    pc.fillStyle = "#fff";
    pc.font = font(700);
    pc.textAlign = "center";
    pc.textBaseline = "middle";
    pc.fillText(ch, 16, 17);
    const d = pc.getImageData(0, 0, 32, 32).data;
    let s = 0;
    for (let i = 3; i < d.length; i += 4) s += d[i];
    measured.push({ ch, cov: s / (32 * 32 * 255) });
  }
  measured.sort((a, b) => a.cov - b.cov);

  // Resample to N glyphs evenly spaced in coverage, then force index 0 blank so
  // empty cells stay empty instead of showing the densest glyph.
  const N = 14;
  const top = measured[measured.length - 1].cov || 1;
  const ramp = [" "];
  for (let i = 1; i < N; i++){
    const want = (i / (N - 1)) * top;
    let best = measured[0], bd = 1e9;
    for (const m of measured){
      const dd = Math.abs(m.cov - want);
      if (dd < bd && !ramp.includes(m.ch)) { bd = dd; best = m; }
    }
    ramp.push(best.ch);
  }

  const CELL = 44;
  const c = document.createElement("canvas");
  c.width = CELL * N; c.height = CELL;
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.fillStyle = "#fff";
  ctx.font = `700 ${Math.round(CELL * 0.86)}px ui-monospace,"DejaVu Sans Mono",Menlo,Consolas,monospace`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (let i = 1; i < N; i++) ctx.fillText(ramp[i], i * CELL + CELL / 2, CELL * 0.53);

  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, c);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  return { tex: t, count: N, ramp: ramp.join("") };
}

/* The wordmark, rasterised at load into a mask the marcher extrudes. Same
 * substance as the rest of the world: it is cells, not type over a picture. */
function buildMark(){
  const W = 768, H = 256;
  const c = document.createElement("canvas");
  c.width = W;
  c.height = H;
  const ctx = c.getContext("2d");
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, W, H);
  ctx.globalCompositeOperation = "lighter";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  // One allocation, one upload, one sample: normal mark, accepted observation,
  // and the final signature all share the same texture.
  ctx.fillStyle = "#f00";
  ctx.font = '600 38px ui-monospace,"DejaVu Sans Mono",Menlo,Consolas,monospace';
  ctx.fillText("IBVOID", W / 2, 58);
  ctx.font = '800 94px ui-monospace,"DejaVu Sans Mono",Menlo,Consolas,monospace';
  ctx.fillText("AF26C13", W / 2, 166);

  ctx.fillStyle = "#0f0";
  ctx.font = '800 162px ui-monospace,"DejaVu Sans Mono",Menlo,Consolas,monospace';
  ctx.fillText(String.fromCharCode(...ANSWER), W / 2, 142);

  // Preserve the signature's aspect ratio inside the packed texture. The
  // public proof and accepted word keep their existing red/green channels;
  // only blue receives these pressure silhouettes and their reveal order.
  const pad = 6;
  const signatureScale = Math.min(
    (W - pad * 2) / SIGNATURE.width,
    (H - pad * 2) / SIGNATURE.height
  );
  const signatureX = (W - SIGNATURE.width * signatureScale) / 2;
  const signatureY = (H - SIGNATURE.height * signatureScale) / 2;
  ctx.save();
  ctx.translate(signatureX, signatureY);
  ctx.scale(signatureScale, signatureScale);
  for (const part of SIGNATURE.parts){
    ctx.fillStyle = `rgb(0, 0, ${part.tone})`;
    ctx.fill(new Path2D(part.d));
  }
  ctx.restore();
  ctx.globalCompositeOperation = "source-over";

  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, c);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return t;
}

function loadTexture(url){
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
                new Uint8Array([12, 12, 12, 255]));
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  const img = new Image();
  img.onload = () => {
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
    reliefReady = true;
    invalidate();
  };
  img.onerror = () => failGL(new Error("relief texture failed to load"));
  img.src = url;
  return t;
}

const atlas = buildAtlas();
const texMark = buildMark();
let reliefReady = false;
const texRel = loadTexture("assets/tex-relief.jpg");

/* --------------------------------------------------------- the framebuffer -- */

const fbo    = gl.createFramebuffer();
const fboTex = gl.createTexture();
gl.bindTexture(gl.TEXTURE_2D, fboTex);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, fboTex, 0);
gl.bindFramebuffer(gl.FRAMEBUFFER, null);

/* Handhelds have real GPUs, so the software-renderer probe above never fires
 * on them — a Pixel reports Adreno, an iPhone reports Apple GPU, and both were
 * getting the desktop budget. They are fill-rate poor and run at DPR 2-3, the
 * exact combination this raymarcher punishes. Start them low and let the
 * governor climb, rather than opening at full cost and stuttering through the
 * first seconds, which is the part of the page that has to land. */
const handheld = (() => {
  const coarse = matchMedia("(pointer: coarse)").matches;
  const narrow = Math.min(screen.width, screen.height) <= 820;
  const thin = (navigator.hardwareConcurrency || 8) <= 6;
  let mobileGpu = false;
  try {
    const info = gl.getExtension("WEBGL_debug_renderer_info");
    const name = info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : "";
    mobileGpu = /adreno|mali|powervr|apple *(a\d|m\d|gpu)|tegra/i.test(name || "");
  } catch { mobileGpu = false; }
  return mobileGpu || (coarse && (narrow || thin));
})();

let dpr = 1, cellPx = 9, gridW = 1, gridH = 1;
let ss = (softwareRenderer || handheld) ? 1 : 2;
let fboW = 1, fboH = 1;
let quality = softwareRenderer ? 0.60 : (handheld ? 0.70 : 1);
let maxSteps = softwareRenderer ? 56 : (handheld ? 72 : MAX_STEPS);

// A handheld at DPR 3 costs 2.25x a DPR 2 desktop for identical CSS pixels,
// and the glyph grid discards that detail anyway.
const dprCap = handheld ? 1.5 : DPR_CAP;
const ceilSteps = maxSteps;
const ceilQuality = quality;

function resize(){
  const cssW = Math.max(1, canvas.clientWidth);
  const cssH = Math.max(1, canvas.clientHeight);
  const pixelCap = Math.sqrt(MAX_CANVAS_PIXELS / (cssW * cssH));
  dpr = Math.max(0.5, Math.min(window.devicePixelRatio || 1, dprCap, pixelCap) * quality);
  const w = Math.max(1, Math.round(cssW * dpr));
  const h = Math.max(1, Math.round(cssH * dpr));
  cellPx = Math.max(4, Math.round(CELL_CSS * dpr));
  const nw = Math.ceil(w / cellPx), nh = Math.ceil(h / cellPx);
  if (canvas.width !== w || canvas.height !== h || nw !== gridW || nh !== gridH){
    canvas.width = w; canvas.height = h;
    gridW = nw; gridH = nh;
    allocScene();
  }
}
function allocScene(){
  fboW = gridW * ss;
  fboH = gridH * ss;
  gl.bindTexture(gl.TEXTURE_2D, fboTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, fboW, fboH, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  if (status !== gl.FRAMEBUFFER_COMPLETE) throw new Error("framebuffer incomplete: " + status);
}

/* ---------------------------------------------------------- choreography -- */

/* Camera per station, plus the frustum shift that keeps the subject clear of the

 * copy. Authored here rather than in the shader because the timing and the
 * framing are design decisions and want to be editable in one place. `shift` is
 * in half-height units and pushes the subject the opposite way, so a negative x
 * moves the world to the right of the screen. */
function camera(w, t, mx, my){
  const e = t * t * (3 - 2 * t);
  switch (w){
    // 00 swings from almost straight down — where a relief map reads as a face
    // — to a low raking angle over the same ground, which is where 01 picks it
    // up. The descent is the station; there is nothing else to look at.
    case 0:
      return { pos: [mx * 10, 96 - e * 52, 26 + e * 52],
               tgt: [0, 5 + e * 5, -4 - e * 8],
               zoom: 1.15 + e * 0.22, shift: [-0.42, 0.02] };
    case 1:                                       // raking over the rows
      return { pos: [mx * 10, 37 - e * 4, 66 - e * 7],
               tgt: [0, 4, 12 - e * 3], zoom: 1.52, shift: [0.38, -0.12] };
    case 2:                                       // pulled back for both lanes
      return { pos: [mx * 8, 46 - e * 5, 74 - e * 8],
               tgt: [0, 3, 0], zoom: 1.34, shift: [-0.34, -0.06] };
    case 3:                                       // trucking past upstream
      return { pos: [-8 + e * 16 + mx * 8, 16 + my * 6, 54 - e * 4],
               tgt: [-8 + e * 16, 14, 0], zoom: 1.62, shift: [0.40, -0.04] };
    case 4:                                       // trucking past the works
      return { pos: [-8 + e * 16 + mx * 9, 18 + my * 7, 54 - e * 4],
               tgt: [-8 + e * 16, 13, 0], zoom: 1.35, shift: [0.42, -0.02] };
    default:                                      // into the mark
      return { pos: [mx * 9, 20 + my * 6, 98 - e * 32],
               tgt: [0, 18, 2], zoom: 1.30, shift: [-0.30, -0.06] };
  }
}

/* Two stations are live through a boundary, so the camera has to be too. */
function blendCam(a, b, k){
  const L = (u, v) => [u[0] + (v[0] - u[0]) * k, u[1] + (v[1] - u[1]) * k,
                       u.length > 2 ? u[2] + (v[2] - u[2]) * k : 0];
  return { pos: L(a.pos, b.pos), tgt: L(a.tgt, b.tgt),
           zoom: a.zoom + (b.zoom - a.zoom) * k,
           shift: [a.shift[0] + (b.shift[0] - a.shift[0]) * k,
                   a.shift[1] + (b.shift[1] - a.shift[1]) * k] };
}

/* ------------------------------------------------------------------ state -- */

const S = {
  act: 0, actTarget: 0,
  after: afterState.target,
  ritual: ritualTarget,
  observe: observeTarget,
  mouse: [0.5, 0.5], mouseSmooth: [0.5, 0.5],
  lens: 0, lensWant: 0,
  motion: still.matches ? 0 : 1,
  focus: -1,
  boot: 0,
  t0: performance.now(),
  frames: 0, fpsAt: performance.now(), fps: 0,
  slowRuns: 0,
  fastRuns: 0,
  lastFrame: performance.now(),
};

let raf = 0;
let pageVisible = !document.hidden;
let frameFailed = false;
let firstFrame = false;

const hud = {
  station: document.getElementById("hud-station"),
  name:    document.getElementById("hud-name"),
  grid:    document.getElementById("hud-grid"),
  fps:     document.getElementById("hud-fps"),
  ramp:    document.getElementById("hud-ramp"),
};
const STATION_NAMES = ["arrival", "measurement", "the split",
                       "upstream", "the works", "proof"];
const ticks = Array.from(document.querySelectorAll(".rail-tick"));

if (hud.ramp) hud.ramp.textContent = atlas.ramp.trim().slice(0, 13);

/* Act position is measured off the sections themselves rather than off a
 * percentage of the document. Sections can then be different heights — the one
 * carrying the table needs more room than the one carrying a sentence — and a
 * station boundary still lands exactly where the copy changes. */
const sections = Array.from(document.querySelectorAll(".station"));
let bands = [];

function measure(){
  let y = 0;
  bands = sections.map(el => {
    const r = el.getBoundingClientRect();
    y = r.top + window.scrollY;
    return { top: y, h: Math.max(1, r.height) };
  });
}

function readScroll(){
  if (!bands.length) return;
  const eye = window.scrollY + window.innerHeight * 0.5;
  let i = bands.length - 1;
  for (let k = 0; k < bands.length; k++){
    if (eye < bands[k].top + bands[k].h){ i = k; break; }
  }
  const local = Math.min(1, Math.max(0, (eye - bands[i].top) / bands[i].h));
  S.actTarget = Math.min(STATIONS - 0.001, i + local);
}
addEventListener("scroll", () => { readScroll(); invalidate(); }, { passive: true });
addEventListener("resize", () => { resize(); measure(); readScroll(); invalidate(); }, { passive: true });

addEventListener("pointermove", event => {
  const r = canvas.getBoundingClientRect();
  S.mouse = [(event.clientX - r.left) / r.width, 1 - (event.clientY - r.top) / r.height];
  invalidate();
}, { passive: true });

function markLensUsed(){
  root.classList.add("lens-used");
}

addEventListener("pointerdown", event => {
  if (event.button !== 0 || ritualState.phase !== "sealed" ||
      afterState.target > 0.001 || isInteractiveEvent(event)) return;
  S.lensWant = 1;
  markLensUsed();
  invalidate();
}, { passive: true });
addEventListener("pointerup", () => {
  S.lensWant = lensLatched ? 1 : 0;
  invalidate();
}, { passive: true });
addEventListener("pointercancel", () => {
  S.lensWant = lensLatched ? 1 : 0;
  invalidate();
}, { passive: true });

const btnLens   = document.getElementById("btn-lens");
const btnMotion = document.getElementById("btn-motion");
let lensLatched = false;

function setLens(on){
  lensLatched = on;
  S.lensWant = on ? 1 : 0;
  if (btnLens) btnLens.setAttribute("aria-pressed", String(on));
  invalidate();
}
function setMotion(on){
  S.motion = on ? 1 : 0;
  root.classList.toggle("no-motion", !on);
  if (btnMotion) btnMotion.setAttribute("aria-pressed", String(on));
  readAfter();
  invalidate();
}

if (btnLens) btnLens.addEventListener("click", () => {
  setLens(!lensLatched);
  markLensUsed();
});
if (btnMotion) btnMotion.addEventListener("click", () => setMotion(!S.motion));

const lensCue = document.getElementById("lens-cue");
if (lensCue) lensCue.addEventListener("click", () => setLens(!lensLatched));

const onMotionPreference = event => setMotion(!event.matches);
if (still.addEventListener) still.addEventListener("change", onMotionPreference);
else if (still.addListener) still.addListener(onMotionPreference);

setMotion(!still.matches);

addEventListener("keydown", e => {
  if (ritualState.phase !== "sealed" || e.metaKey || e.ctrlKey || e.altKey ||
      isInteractiveEvent(e)) return;
  if (e.key === "l" || e.key === "L"){ setLens(!lensLatched); }
  if (e.key === "m" || e.key === "M"){ setMotion(!S.motion); }
});

// Hovering or focusing a work lights its object in the scene. The list and the
// world are the same data; keeping them wired is cheap and the link is the joke.
document.querySelectorAll("[data-work]").forEach(el => {
  const i = parseInt(el.dataset.work, 10);
  const on  = () => { S.focus = i; invalidate(); };
  const off = () => { S.focus = -1; invalidate(); };
  el.addEventListener("pointerenter", on);
  el.addEventListener("pointerleave", off);
  el.addEventListener("focusin", on);
  el.addEventListener("focusout", off);
});

/* ------------------------------------------------------------------ frame -- */

const damp = (a, b, rate, dt) => a + (b - a) * (1 - Math.exp(-rate * dt));

function failGL(error){
  if (frameFailed) return;
  frameFailed = true;
  if (raf) cancelAnimationFrame(raf);
  raf = 0;
  root.classList.remove("gl-ready");
  root.classList.add("gl-failed");
  console.error(error);
}

function invalidate(){
  if (!raf && pageVisible && !frameFailed) raf = requestAnimationFrame(draw);
}
afterState.onchange = invalidate;

function draw(now){
  raf = 0;
  if (frameFailed || !pageVisible) return;

  try {
    resize();

    const dt = Math.min(0.05, Math.max(0.001, (now - S.lastFrame) / 1000));
    S.lastFrame = now;
    const time = (now - S.t0) / 1000;
    const reduced = S.motion < 0.5;

    S.boot = reduced ? 1 : Math.min(1, time / 2.8);
    S.act = reduced ? S.actTarget : damp(S.act, S.actTarget, 7.2, dt);
    S.after = reduced ? afterState.target : damp(S.after, afterState.target, 8.4, dt);
    S.ritual = reduced ? ritualTarget : damp(S.ritual, ritualTarget, 5.8, dt);
    S.observe = reduced ? observeTarget : damp(S.observe, observeTarget, 12.0, dt);
    S.mouseSmooth[0] = reduced ? S.mouse[0] : damp(S.mouseSmooth[0], S.mouse[0], 8.0, dt);
    S.mouseSmooth[1] = reduced ? S.mouse[1] : damp(S.mouseSmooth[1], S.mouse[1], 8.0, dt);
    S.lens = damp(S.lens, S.lensWant, 12.0, dt);

    const wi = Math.min(STATIONS - 1, Math.floor(S.act));
    const splitProgress = smooth(2.04, 2.76, S.act);
    const boundary = Math.round(S.act);
    let wa = wi, wb = wi, morph = 0;
    if (boundary >= 1 && boundary <= STATIONS - 1 &&
        Math.abs(S.act - boundary) < BAND){
      wa = boundary - 1;
      wb = boundary;
      morph = (S.act - (boundary - BAND)) / (2 * BAND);
    }

    const A = WORLDS[wa], B = WORLDS[wb];
    const uMin = [Math.min(A.min[0], B.min[0]), Math.min(A.min[1], B.min[1]),
                  Math.min(A.min[2], B.min[2])];
    const uMax = [Math.max(A.max[0], B.max[0]), Math.max(A.max[1], B.max[1]),
                  Math.max(A.max[2], B.max[2])];

    const mx = reduced ? 0 : (S.mouseSmooth[0] - 0.5) * 2;
    const my = reduced ? 0 : (S.mouseSmooth[1] - 0.5) * 2;
    const intro = reduced ? 1 : smooth(0.08, 0.92, S.boot);
    const localA = wa === 0 ? Math.max(S.act - wa, intro * 0.68) : S.act - wa;
    let cam = wa === wb
      ? camera(wa, localA, mx, my)
      : blendCam(camera(wa, Math.min(1, localA), mx, my),
                 camera(wb, Math.max(0, S.act - wb), mx, my),
                 smooth(0, 1, morph));
    if (wi === 5 && S.ritual > 0.001){
      const ritualMix = smooth(0.02, 0.96, S.ritual);
      const cupCam = {
        pos: [18 + mx * 5, 30 + my * 3, 82],
        tgt: [-2, 10, 0],
        zoom: 1.34,
        shift: [0.29, -0.02],
      };
      cam = blendCam(cam, cupCam, ritualMix);
      if (S.after > 0.001){
        const dive = smooth(0.13, 0.30, S.after);
        const ringTurn = smooth(0.22, 0.48, S.after);
        const ringCenter = -14 * (1 - ringTurn);
        const overheadCam = {
          pos: [ringCenter + mx * 0.8, 66, 1 + my * 0.8],
          tgt: [ringCenter, 1, 0],
          zoom: 1.38,
          shift: [0, 0],
        };
        cam = blendCam(cam, overheadCam, dive);

        const breach = smooth(0.29, 0.58, S.after);
        const wallCam = {
          pos: [mx * 3, 21 + my * 2, 72 - breach * 7],
          tgt: [0, 18, 0],
          zoom: 1.46,
          shift: [0, -0.04],
        };
        cam = blendCam(cam, wallCam, breach);
      }
    }
    if (innerWidth < 900) cam.shift = [0, 0.10];

    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.viewport(0, 0, fboW, fboH);
    gl.useProgram(progScene);
    gl.uniform2f(uS.res, fboW, fboH);
    gl.uniform2fv(uS.shift, cam.shift);
    gl.uniform3fv(uS.camPos, cam.pos);
    gl.uniform3fv(uS.camTgt, cam.tgt);
    gl.uniform3fv(uS.uMin, uMin);
    gl.uniform3fv(uS.uMax, uMax);
    gl.uniform1f(uS.time, reduced ? 4.0 : time);
    gl.uniform1f(uS.worldA, wa);
    gl.uniform1f(uS.worldB, wb);
    gl.uniform1f(uS.morph, morph);
    gl.uniform1f(uS.phase, reduced ? 0 : time * 2.2);
    gl.uniform1f(uS.cut, 0);
    gl.uniform1f(uS.focus, S.focus);
    gl.uniform1f(uS.inflate, wi === 0 ? (reduced ? 1 : smooth(0.08, 0.94, S.boot)) : 1);
    gl.uniform1f(uS.split, splitProgress);
    gl.uniform1f(uS.zoom, cam.zoom);
    gl.uniform1f(uS.maxSteps, maxSteps);
    gl.uniform1f(uS.after, S.after);
    gl.uniform1f(uS.ritual, S.ritual);
    gl.uniform1f(uS.observe, S.observe);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texRel);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, texMark);
    gl.uniform1i(uS.relief, 0);
    gl.uniform1i(uS.mark, 1);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.useProgram(progAscii);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, fboTex);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, atlas.tex);
    gl.uniform1i(uA.scene, 0);
    gl.uniform1i(uA.atlas, 1);
    gl.uniform2f(uA.res, canvas.width, canvas.height);
    gl.uniform2f(uA.grid, gridW, gridH);
    gl.uniform2f(uA.mouse, S.mouseSmooth[0] * canvas.width, S.mouseSmooth[1] * canvas.height);
    gl.uniform1f(uA.cell, cellPx);
    gl.uniform1f(uA.count, atlas.count);
    gl.uniform1f(uA.time, reduced ? 0 : time);
    gl.uniform1f(uA.boot, S.boot);
    gl.uniform1f(uA.lens, S.lens *
      (1 - smooth(0.08, 0.24, Math.max(S.after, S.ritual))));
    gl.uniform1f(uA.lensR, Math.min(canvas.width, canvas.height) * 0.19);
    gl.uniform1f(uA.motion, S.motion);
    gl.uniform1f(uA.warpK, reduced ? 0 : 0.50);
    gl.uniform1f(uA.reveal, reduced ? 0 : 1.0 - smooth(0.18, 0.88, S.boot));
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    if (!firstFrame && reliefReady){
      firstFrame = true;
      root.classList.add("gl-ready");
    }

    S.frames++;
    if (now - S.fpsAt > 350){
      S.fps = Math.round(S.frames * 1000 / (now - S.fpsAt));
      S.frames = 0;
      S.fpsAt = now;
      if (hud.fps) hud.fps.textContent = S.fps + " fps";

      // Shed load in order of what it costs to look at: march distance first
      // (far detail only), then supersampling, and resolution last — blurring
      // the whole field is the most visible thing available, so it goes last.
      if (S.fps < 50){
        if (++S.slowRuns >= 2){
          S.slowRuns = 0;
          if (maxSteps > MIN_STEPS){
            maxSteps = Math.max(MIN_STEPS, maxSteps - 16);
          } else if (ss === 2){
            ss = 1;
            allocScene();
          } else if (quality > MIN_QUALITY){
            quality = Math.max(MIN_QUALITY, quality - 0.15);
            resize();
          }
        }
      } else {
        S.slowRuns = 0;
        // Climb back once there is real headroom, so one hitch during a heavy
        // station does not pin the page at its floor for the rest of the visit.
        if (S.fps >= 58 && ++S.fastRuns >= 8){
          S.fastRuns = 0;
          if (maxSteps < ceilSteps){
            maxSteps = Math.min(ceilSteps, maxSteps + 16);
          } else if (quality < ceilQuality){
            quality = Math.min(ceilQuality, quality + 0.15);
            resize();
          }
        }
      }
      if (hud.grid) {
        hud.grid.textContent = gridW + "×" + gridH + " · ss " + ss +
          "× · q " + quality.toFixed(2) + " · " + maxSteps + " steps";
      }
    }

    const chapter = Math.max(0, Math.min(STATIONS - 1,
      Math.round(S.act - 0.5 + 0.001)));
    if (hud.station && hud.station.dataset.i !== String(chapter)){
      hud.station.dataset.i = String(chapter);
      hud.station.textContent = String(chapter).padStart(2, "0");
      if (hud.name) hud.name.textContent = STATION_NAMES[chapter];
      root.dataset.station = String(chapter);
      sections.forEach((section, i) => section.classList.toggle("is-active", i === chapter));
      ticks.forEach((tick, i) => {
        tick.setAttribute("aria-current", i === chapter ? "true" : "false");
      });
    }

    const unsettled = Math.abs(S.act - S.actTarget) > 0.001 ||
                      Math.abs(S.after - afterState.target) > 0.001 ||
                      Math.abs(S.ritual - ritualTarget) > 0.001 ||
                      Math.abs(S.mouseSmooth[0] - S.mouse[0]) > 0.001 ||
                      Math.abs(S.mouseSmooth[1] - S.mouse[1]) > 0.001 ||
                      Math.abs(S.lens - S.lensWant) > 0.001 ||
                      S.boot < 1;
    if (S.motion > 0.5 || unsettled) invalidate();
  } catch (error) {
    failGL(error);
  }
}

function smooth(a, b, x){
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

try {
  resize();
  measure();
  readScroll();
  invalidate();
} catch (error) {
  failGL(error);
}

addEventListener("load", () => {
  measure();
  readScroll();
  invalidate();
});

document.addEventListener("visibilitychange", () => {
  pageVisible = !document.hidden;
  S.lastFrame = performance.now();
  if (pageVisible) invalidate();
});

canvas.addEventListener("webglcontextlost", event => {
  event.preventDefault();
  frameFailed = true;
  root.classList.remove("gl-ready");
  root.classList.add("gl-failed");
});

canvas.addEventListener("webglcontextrestored", () => location.reload());


/* ------------------------------------------------- orientation hint -- */
/* Shown only where turning the device is actually an option: a touch screen
 * held upright. It is advice, not a gate — the page is fully usable in
 * portrait, so this never covers the content or blocks scrolling, and it
 * retires itself as soon as the phone turns. */
(() => {
  const hint = document.getElementById("rotate-hint");
  if (!hint) return;
  const dismiss = document.getElementById("rotate-dismiss");
  const portrait = matchMedia("(orientation: portrait)");
  const touch = matchMedia("(pointer: coarse)");

  let retired = false;
  try { retired = sessionStorage.getItem("ibvoid.rotate") === "off"; } catch {}

  let timer = 0;
  const hide = () => { hint.hidden = true; clearTimeout(timer); };
  const retire = () => {
    retired = true;
    hide();
    try { sessionStorage.setItem("ibvoid.rotate", "off"); } catch {}
  };

  function sync(){
    if (retired || !touch.matches || !portrait.matches){ hide(); return; }
    if (!hint.hidden) return;
    hint.hidden = false;
    // Say it once and get out of the way; a hint that never leaves is a nag.
    clearTimeout(timer);
    timer = setTimeout(hide, 7000);
  }

  dismiss?.addEventListener("click", retire);
  // Turning the phone is the hint being taken, so stop offering it.
  portrait.addEventListener?.("change", () => {
    if (!portrait.matches) retire(); else sync();
  });
  touch.addEventListener?.("change", sync);
  addEventListener("scroll", () => { if (!hint.hidden) hide(); },
                   { passive: true, once: true });

  sync();
})();

})();
