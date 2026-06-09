// Blokus Duo engine (browser/Node ES module). Mirrors the tested Python engine
// in blokus_core/: 14x14 board, start cells (4,4)/(9,9), 21 pieces -> 91
// orientations, anchor-based legal move generation, scoring.

export const BOARD = 14;
export const NCELL = BOARD * BOARD;            // 196
export const START = [[4, 4], [9, 9]];          // player 0, player 1 (zero-indexed)
export const START_IDX = START.map(([r, c]) => r * BOARD + c);
export const PASS = -1;

const ORTHO = [[-1, 0], [1, 0], [0, -1], [0, 1]];
const DIAG = [[-1, -1], [-1, 1], [1, -1], [1, 1]];

// Base shapes (standard polyomino names); all 8 D4 images are generated + deduped.
const BASE = [
  ["I1", [[0, 0]]],
  ["I2", [[0, 0], [0, 1]]],
  ["I3", [[0, 0], [0, 1], [0, 2]]],
  ["V3", [[0, 0], [1, 0], [1, 1]]],
  ["I4", [[0, 0], [0, 1], [0, 2], [0, 3]]],
  ["O4", [[0, 0], [0, 1], [1, 0], [1, 1]]],
  ["T4", [[0, 0], [0, 1], [0, 2], [1, 1]]],
  ["S4", [[0, 1], [0, 2], [1, 0], [1, 1]]],
  ["L4", [[0, 0], [1, 0], [2, 0], [2, 1]]],
  ["F5", [[0, 1], [0, 2], [1, 0], [1, 1], [2, 1]]],
  ["I5", [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4]]],
  ["L5", [[0, 0], [1, 0], [2, 0], [3, 0], [3, 1]]],
  ["N5", [[0, 1], [1, 1], [2, 0], [2, 1], [3, 0]]],
  ["P5", [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0]]],
  ["T5", [[0, 0], [0, 1], [0, 2], [1, 1], [2, 1]]],
  ["U5", [[0, 0], [0, 2], [1, 0], [1, 1], [1, 2]]],
  ["V5", [[0, 0], [1, 0], [2, 0], [2, 1], [2, 2]]],
  ["W5", [[0, 0], [1, 0], [1, 1], [2, 1], [2, 2]]],
  ["X5", [[0, 1], [1, 0], [1, 1], [1, 2], [2, 1]]],
  ["Y5", [[0, 1], [1, 0], [1, 1], [2, 1], [3, 1]]],
  ["Z5", [[0, 0], [0, 1], [1, 1], [2, 1], [2, 2]]],
];
export const PIECE_NAMES = BASE.map((b) => b[0]);
export const NPIECE = BASE.length;              // 21

const norm = (cells) => {
  const mr = Math.min(...cells.map((c) => c[0]));
  const mc = Math.min(...cells.map((c) => c[1]));
  return cells.map(([r, c]) => [r - mr, c - mc]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
};
const rot = (cells) => cells.map(([r, c]) => [c, -r]);
const refl = (cells) => cells.map(([r, c]) => [r, -c]);
const keyOf = (cells) => cells.map((c) => c.join(",")).join(";");

function orientations(shape) {
  const seen = new Set(); const out = [];
  let cur = shape;
  for (let i = 0; i < 4; i++) {
    for (const img of [cur, refl(cur)]) {
      const n = norm(img); const k = keyOf(n);
      if (!seen.has(k)) { seen.add(k); out.push(n); }
    }
    cur = rot(cur);
  }
  return out;
}

export const VARIANTS = [];        // variant id -> normalized cell list
export const VARIANT_PIECE = [];   // variant id -> piece id
export const PIECE_VARIANTS = [];  // piece id -> [variant ids]
export const PIECE_SIZE = [];
for (let pid = 0; pid < NPIECE; pid++) {
  const sh = norm(BASE[pid][1]);
  PIECE_SIZE.push(sh.length);
  const ids = [];
  for (const o of orientations(sh)) { ids.push(VARIANTS.length); VARIANTS.push(o); VARIANT_PIECE.push(pid); }
  PIECE_VARIANTS.push(ids);
}
export const NVARIANT = VARIANTS.length;        // 91
export const ACTION_SPACE = NVARIANT * NCELL;   // 17836

// Rotate/flip transition tables (variant -> variant), for the UI buttons.
export const ROT_OF = []; export const FLIP_OF = [];
{
  const k2v = new Map();
  VARIANTS.forEach((v, i) => k2v.set(keyOf(v), i));
  VARIANTS.forEach((v) => { ROT_OF.push(k2v.get(keyOf(norm(rot(v))))); FLIP_OF.push(k2v.get(keyOf(norm(refl(v))))); });
}

export const decode = (a) => [Math.floor(a / NCELL), a % NCELL];
export const encode = (v, refCell) => v * NCELL + refCell;
export function placementCells(a) {
  const [v, rc] = decode(a); const rr = Math.floor(rc / BOARD); const cc = rc % BOARD;
  return VARIANTS[v].map(([r, c]) => [rr + r, cc + c]);
}
const onBoard = (r, c) => r >= 0 && r < BOARD && c >= 0 && c < BOARD;

export function initialState(first = 0) {
  return {
    occ: [new Uint8Array(NCELL), new Uint8Array(NCELL)],
    used: [new Array(NPIECE).fill(false), new Array(NPIECE).fill(false)],
    last: [-1, -1], current: first, finished: [false, false], moves: 0,
  };
}
export function cloneState(s) {
  return {
    occ: [s.occ[0].slice(), s.occ[1].slice()],
    used: [s.used[0].slice(), s.used[1].slice()],
    last: [...s.last], current: s.current, finished: [...s.finished], moves: s.moves,
  };
}

function forbiddenMask(own, opp) {
  const f = new Uint8Array(NCELL);
  for (let i = 0; i < NCELL; i++) if (own[i] || opp[i]) f[i] = 1;
  for (let i = 0; i < NCELL; i++) if (own[i]) {
    const r = (i / BOARD) | 0; const c = i % BOARD;
    for (const [dr, dc] of ORTHO) { const nr = r + dr; const nc = c + dc; if (onBoard(nr, nc)) f[nr * BOARD + nc] = 1; }
  }
  return f;
}

function anchorList(own, opp, startIdx, forbidden) {
  if (!own.some((x) => x)) return forbidden[startIdx] ? [] : [startIdx];
  const isA = new Uint8Array(NCELL);
  for (let i = 0; i < NCELL; i++) if (own[i]) {
    const r = (i / BOARD) | 0; const c = i % BOARD;
    for (const [dr, dc] of DIAG) { const nr = r + dr; const nc = c + dc; if (onBoard(nr, nc)) { const j = nr * BOARD + nc; if (!forbidden[j]) isA[j] = 1; } }
  }
  const out = []; for (let i = 0; i < NCELL; i++) if (isA[i]) out.push(i); return out;
}

function legalCore(own, opp, used, startIdx) {
  const forbidden = forbiddenMask(own, opp);
  const anchors = anchorList(own, opp, startIdx, forbidden);
  if (anchors.length === 0) return [];
  const avail = [];
  for (let pid = 0; pid < NPIECE; pid++) if (!used[pid]) for (const v of PIECE_VARIANTS[pid]) avail.push(v);
  const res = new Set();
  for (const a of anchors) {
    const ar = (a / BOARD) | 0; const ac = a % BOARD;
    for (const v of avail) {
      const offs = VARIANTS[v];
      for (const [or, oc] of offs) {
        const refr = ar - or; const refc = ac - oc;
        if (refr < 0 || refc < 0) continue;
        let ok = true;
        for (const [qr, qc] of offs) {
          const R = refr + qr; const C = refc + qc;
          if (R >= BOARD || C >= BOARD) { ok = false; break; }
          if (forbidden[R * BOARD + C]) { ok = false; break; }
        }
        if (ok) res.add(v * NCELL + (refr * BOARD + refc));
      }
    }
  }
  return [...res];
}

export const legalActions = (s) => legalCore(s.occ[s.current], s.occ[1 - s.current], s.used[s.current], START_IDX[s.current]);
export const hasLegal = (s, p) => legalCore(s.occ[p], s.occ[1 - p], s.used[p], START_IDX[p]).length > 0;
export const isTerminal = (s) => !hasLegal(s, 0) && !hasLegal(s, 1);

// Number of anchor (corner-liberty) cells for a player — used by the heuristic AI.
export function anchorCount(own, opp, startIdx) {
  return anchorList(own, opp, startIdx, forbiddenMask(own, opp)).length;
}

export function applyAction(s, a) {
  const n = cloneState(s); const p = n.current;
  if (a === PASS) { n.finished[p] = true; }
  else {
    const [v] = decode(a);
    for (const [r, c] of placementCells(a)) n.occ[p][r * BOARD + c] = 1;
    n.used[p][VARIANT_PIECE[v]] = true; n.last[p] = VARIANT_PIECE[v];
  }
  n.moves++; n.current = 1 - p; return n;
}

export function squaresPlaced(s, p) {
  let t = 0; for (let k = 0; k < NPIECE; k++) if (s.used[p][k]) t += PIECE_SIZE[k]; return t;
}
// Blokus score in the familiar "negative remaining" convention (starts at -89).
export function score(s, p) {
  const placed = squaresPlaced(s, p);
  if (placed === 89) return s.last[p] === 0 ? 20 : 15;   // all placed: +15, +5 if monomino last
  return placed - 89;
}
export function outcome(s) { const a = score(s, 0); const b = score(s, 1); return a > b ? 1 : a < b ? -1 : 0; }

// Legal reference cells for a given variant (where its bbox origin can land).
export function legalRefsForVariant(s, variantId) {
  const refs = new Set();
  for (const a of legalActions(s)) { const [v, rc] = decode(a); if (v === variantId) refs.add(rc); }
  return refs;
}
