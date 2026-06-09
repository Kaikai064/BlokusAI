// Validate the JS engine against an independent brute-force generator.
// Run: node docs/engine.test.mjs
import * as E from "./engine.js";

let fails = 0;
const ok = (c, m) => { if (!c) { console.error("FAIL:", m); fails++; } };

ok(E.NVARIANT === 91, `NVARIANT=${E.NVARIANT} (expected 91)`);
ok(E.NPIECE === 21, "NPIECE");
ok(JSON.stringify(E.PIECE_SIZE) ===
   JSON.stringify([1, 2, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]), "PIECE_SIZE");
ok(E.ACTION_SPACE === 91 * 196, "ACTION_SPACE");
ok(E.ROT_OF.every((x) => x !== undefined) && E.FLIP_OF.every((x) => x !== undefined),
   "rotate/flip transition tables complete");

const N = 14;
function brute(own, opp, used, startIdx) {
  const occ = (i) => own[i] || opp[i];
  const ortho = new Uint8Array(196); const diag = new Uint8Array(196);
  for (let i = 0; i < 196; i++) if (own[i]) {
    const r = (i / N) | 0; const c = i % N;
    for (const [dr, dc] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) { const nr = r + dr; const nc = c + dc; if (nr >= 0 && nr < N && nc >= 0 && nc < N) ortho[nr * N + nc] = 1; }
    for (const [dr, dc] of [[-1, -1], [-1, 1], [1, -1], [1, 1]]) { const nr = r + dr; const nc = c + dc; if (nr >= 0 && nr < N && nc >= 0 && nc < N) diag[nr * N + nc] = 1; }
  }
  const first = !own.some((x) => x);
  const res = new Set();
  for (let pid = 0; pid < E.NPIECE; pid++) {
    if (used[pid]) continue;
    for (const v of E.PIECE_VARIANTS[pid]) {
      const offs = E.VARIANTS[v];
      const maxr = Math.max(...offs.map((o) => o[0])); const maxc = Math.max(...offs.map((o) => o[1]));
      for (let rr = 0; rr < N - maxr; rr++) for (let cc = 0; cc < N - maxc; cc++) {
        let overlap = false; let oOrtho = false; let oDiag = false; let coversStart = false;
        for (const [r, c] of offs) { const idx = (rr + r) * N + (cc + c); if (occ(idx)) overlap = true; if (ortho[idx]) oOrtho = true; if (diag[idx]) oDiag = true; if (idx === startIdx) coversStart = true; }
        if (overlap) continue;
        if (first) { if (coversStart) res.add(v * 196 + rr * N + cc); }
        else { if (oOrtho || !oDiag) continue; res.add(v * 196 + rr * N + cc); }
      }
    }
  }
  return res;
}

{
  const s = E.initialState();
  const la = E.legalActions(s);
  ok(la.length > 0, "first move has legal moves");
  ok(la.every((a) => E.placementCells(a).some(([r, c]) => r === 4 && c === 4)), "first moves cover (4,4)");
  const b = brute(s.occ[0], s.occ[1], s.used[0], E.START_IDX[0]);
  ok(la.length === b.size && la.every((a) => b.has(a)), `first-move fast=${la.length} brute=${b.size}`);
  console.log("legal first moves:", la.length);
}

function rngF(seed) { let x = (seed >>> 0) || 1; return () => { x ^= x << 13; x ^= x >>> 17; x ^= x << 5; x >>>= 0; return x / 4294967296; }; }
for (let seed = 1; seed <= 4; seed++) {
  const rng = rngF(seed); let s = E.initialState(seed % 2); let cp = 0; let ply = 0;
  while (cp < 2 && ply < 120) {
    const p = s.current;
    const fast = new Set(E.legalActions(s));
    const b = brute(s.occ[p], s.occ[1 - p], s.used[p], E.START_IDX[p]);
    ok(fast.size === b.size && [...fast].every((a) => b.has(a)), `fuzz seed=${seed} ply=${ply} fast=${fast.size} brute=${b.size}`);
    const acts = [...b];
    if (acts.length) { s = E.applyAction(s, acts[(rng() * acts.length) | 0]); cp = 0; }
    else { s = E.applyAction(s, E.PASS); cp++; }
    ply++;
  }
  ok(E.isTerminal(s), `seed=${seed} reaches terminal`);
  ok([-1, 0, 1].includes(E.outcome(s)), `seed=${seed} outcome valid`);
}

console.log(fails === 0 ? "ALL ENGINE TESTS PASSED" : `${fails} FAILURES`);
process.exit(fails ? 1 : 0);
