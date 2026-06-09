// Built-in JavaScript opponents (no neural net). Good enough for casual play;
// the strong ONNX/neural AI is a later phase.
import * as E from "./engine.js";

const pieceSize = (a) => E.PIECE_SIZE[E.VARIANT_PIECE[Math.floor(a / E.NCELL)]];

// level: 0 = random, 1 = greedy (biggest piece), 2 = blocking (deny opponent).
export function aiMove(s, level = 2, rnd = Math.random) {
  const acts = E.legalActions(s);
  if (acts.length === 0) return E.PASS;
  if (level === 0) return acts[Math.floor(rnd() * acts.length)];

  const p = s.current;
  let best = -Infinity; let top = [];
  for (const a of acts) {
    let sc;
    if (level === 1) {
      sc = pieceSize(a);
    } else {
      const ns = E.applyAction(s, a);
      const ownAnc = E.anchorCount(ns.occ[p], ns.occ[1 - p], E.START_IDX[p]);
      const oppAnc = E.anchorCount(ns.occ[1 - p], ns.occ[p], E.START_IDX[1 - p]);
      sc = 2 * pieceSize(a) + ownAnc - oppAnc;
    }
    if (sc > best) { best = sc; top = [a]; }
    else if (sc === best) top.push(a);
  }
  return top[Math.floor(rnd() * top.length)];
}
