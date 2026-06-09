// Blokus Duo — interactive UI. You play Black (moves first); the computer plays
// White. Pick a piece, Rotate/Flip it, and the green dots show every legal
// starting cell — hover to preview the footprint, click to place.
import * as E from "./engine.js";
import { aiMove } from "./ai.js";

const HUMAN = 0, AI = 1;
const CELL = 36;
const COLORS = {
  empty: "#b9cdbb", grid: "#9fb6a1",
  p0: "#2f3240", p0edge: "#191b22",
  p1: "#f3ead2", p1edge: "#c7b98f",
  start: "#7d9c80", dot: "#2f9e5e",
  okFill: "rgba(47,158,94,0.55)", badFill: "rgba(201,72,72,0.5)",
};

let state = E.initialState(HUMAN);
let sel = { pid: null, variant: null };
let legalRefs = new Set();
let hoverIdx = null;
let aiLevel = 2;
let busy = false;

const $ = (id) => document.getElementById(id);
const canvas = $("board");
const ctx = canvas.getContext("2d");
canvas.width = E.BOARD * CELL;
canvas.height = E.BOARD * CELL;

// ---- piece mini-renderer -------------------------------------------------
function miniSVG(cells, color, edge) {
  const maxr = Math.max(...cells.map((c) => c[0]));
  const maxc = Math.max(...cells.map((c) => c[1]));
  const u = 9, pad = 2;
  const w = (maxc + 1) * u + pad * 2, h = (maxr + 1) * u + pad * 2;
  let rects = "";
  for (const [r, c] of cells) {
    rects += `<rect x="${pad + c * u}" y="${pad + r * u}" width="${u - 1}" height="${u - 1}" rx="1.5" fill="${color}" stroke="${edge}" stroke-width="0.6"/>`;
  }
  return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">${rects}</svg>`;
}

// ---- panels --------------------------------------------------------------
function buildPanels() {
  for (const player of [HUMAN, AI]) {
    const panel = $(player === HUMAN ? "leftPanel" : "rightPanel");
    const isHuman = player === HUMAN;
    panel.innerHTML = `
      <div class="phead">
        <span class="dot ${isHuman ? "black" : "white"}"></span>
        <span class="pname">${isHuman ? "You (Black)" : "Computer (White)"}</span>
        <span class="pscore" id="score${player}"></span>
      </div>
      <div class="pmeta"><span id="left${player}"></span><span>Start (${E.START[player][0]},${E.START[player][1]})</span></div>
      <div class="pieces" id="tray${player}"></div>`;
    const tray = $("tray" + player);
    for (let pid = 0; pid < E.NPIECE; pid++) {
      const card = document.createElement("div");
      card.className = "card";
      card.dataset.pid = pid;
      card.dataset.player = player;
      card.innerHTML = `<div class="mini"></div><div class="cname">${E.PIECE_NAMES[pid]}</div>`;
      if (isHuman) card.addEventListener("click", () => selectPiece(pid));
      tray.appendChild(card);
    }
  }
}

function refreshTrays() {
  for (const player of [HUMAN, AI]) {
    const color = player === HUMAN ? COLORS.p0 : COLORS.p1;
    const edge = player === HUMAN ? COLORS.p0edge : COLORS.p1edge;
    $("score" + player).textContent = E.score(state, player);
    $("left" + player).textContent = `${state.used[player].filter((u) => !u).length} pieces left`;
    document.querySelectorAll(`#tray${player} .card`).forEach((card) => {
      const pid = +card.dataset.pid;
      const used = state.used[player][pid];
      card.classList.toggle("used", used);
      const showVariant = (player === HUMAN && sel.pid === pid && sel.variant != null)
        ? sel.variant : E.PIECE_VARIANTS[pid][0];
      card.querySelector(".mini").innerHTML = miniSVG(E.VARIANTS[showVariant], used ? "#cdd6ce" : color, used ? "#cdd6ce" : edge);
      card.classList.toggle("selected", player === HUMAN && sel.pid === pid);
    });
  }
}

// ---- board drawing -------------------------------------------------------
function cellRect(idx) { const r = (idx / E.BOARD) | 0; const c = idx % E.BOARD; return [c * CELL, r * CELL]; }

function drawCell(idx, fill, edge) {
  const [x, y] = cellRect(idx);
  ctx.fillStyle = fill;
  ctx.fillRect(x + 1, y + 1, CELL - 2, CELL - 2);
  if (edge) { ctx.strokeStyle = edge; ctx.lineWidth = 2; ctx.strokeRect(x + 2, y + 2, CELL - 4, CELL - 4); }
}

function drawBoard() {
  ctx.fillStyle = COLORS.empty;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  // start-cell markers (when empty)
  for (const player of [HUMAN, AI]) {
    const si = E.START_IDX[player];
    if (!state.occ[0][si] && !state.occ[1][si]) {
      const [x, y] = cellRect(si);
      ctx.fillStyle = COLORS.start;
      ctx.beginPath(); ctx.arc(x + CELL / 2, y + CELL / 2, 5, 0, 7); ctx.fill();
    }
  }
  // placed pieces
  for (let i = 0; i < E.NCELL; i++) {
    if (state.occ[0][i]) drawCell(i, COLORS.p0, COLORS.p0edge);
    else if (state.occ[1][i]) drawCell(i, COLORS.p1, COLORS.p1edge);
  }
  // legal-start dots for the selected piece
  if (sel.variant != null && state.current === HUMAN) {
    ctx.fillStyle = COLORS.dot;
    for (const ref of legalRefs) {
      const [x, y] = cellRect(ref);
      ctx.beginPath(); ctx.arc(x + CELL / 2, y + CELL / 2, 4, 0, 7); ctx.fill();
    }
  }
  // hover footprint
  if (hoverIdx != null && sel.variant != null && state.current === HUMAN) {
    const ok = legalRefs.has(hoverIdx);
    const rr = (hoverIdx / E.BOARD) | 0; const cc = hoverIdx % E.BOARD;
    ctx.fillStyle = ok ? COLORS.okFill : COLORS.badFill;
    for (const [dr, dc] of E.VARIANTS[sel.variant]) {
      const R = rr + dr; const C = cc + dc;
      if (R >= 0 && R < E.BOARD && C >= 0 && C < E.BOARD) ctx.fillRect(C * CELL + 1, R * CELL + 1, CELL - 2, CELL - 2);
    }
  }
  // grid
  ctx.strokeStyle = COLORS.grid; ctx.lineWidth = 1;
  for (let k = 0; k <= E.BOARD; k++) {
    ctx.beginPath(); ctx.moveTo(k * CELL, 0); ctx.lineTo(k * CELL, canvas.height); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, k * CELL); ctx.lineTo(canvas.width, k * CELL); ctx.stroke();
  }
}

// ---- status / turn -------------------------------------------------------
function render() {
  drawBoard();
  refreshTrays();
  const legal = E.legalActions(state).length;
  $("legalCount").textContent = `${legal} legal move${legal === 1 ? "" : "s"}`;
  $("pass").disabled = !(state.current === HUMAN && legal === 0 && !E.isTerminal(state));
  if (E.isTerminal(state)) {
    const o = E.outcome(state);
    $("turn").textContent = o > 0 ? "You win! 🎉" : o < 0 ? "Computer wins" : "Draw";
    $("status").textContent = `Final — You ${E.score(state, 0)}  ·  Computer ${E.score(state, 1)}`;
  } else if (busy) {
    $("turn").textContent = "Computer thinking…";
    $("status").textContent = "";
  } else {
    $("turn").textContent = "Your turn (Black)";
    $("status").textContent = sel.variant != null
      ? "Click a green-dotted cell to place (hover to preview). Rotate/Flip to reorient."
      : (legal === 0 ? "No legal moves — press Pass." : "Pick a piece on the left.");
  }
}

// ---- interaction ---------------------------------------------------------
function selectPiece(pid) {
  if (busy || state.current !== HUMAN || state.used[HUMAN][pid]) return;
  sel = { pid, variant: E.PIECE_VARIANTS[pid][0] };
  legalRefs = E.legalRefsForVariant(state, sel.variant);
  render();
}
function reorient(map) {
  if (sel.variant == null) return;
  sel.variant = map[sel.variant];
  legalRefs = E.legalRefsForVariant(state, sel.variant);
  render();
}
function pixelToIdx(ev) {
  const rect = canvas.getBoundingClientRect();
  const x = (ev.clientX - rect.left) * (canvas.width / rect.width);
  const y = (ev.clientY - rect.top) * (canvas.height / rect.height);
  const c = Math.min(E.BOARD - 1, Math.max(0, (x / CELL) | 0));
  const r = Math.min(E.BOARD - 1, Math.max(0, (y / CELL) | 0));
  return r * E.BOARD + c;
}

function place(refIdx) {
  if (busy || state.current !== HUMAN || sel.variant == null || !legalRefs.has(refIdx)) return;
  state = E.applyAction(state, E.encode(sel.variant, refIdx));
  sel = { pid: null, variant: null }; legalRefs = new Set(); hoverIdx = null;
  afterMove();
}

function afterMove() {
  render();
  if (E.isTerminal(state)) return;
  if (state.current === AI) { busy = true; render(); setTimeout(stepAI, 350); }
  else if (!E.hasLegal(state, HUMAN)) { busy = true; render(); setTimeout(() => { state = E.applyAction(state, E.PASS); busy = false; afterMove(); }, 350); }
}
function stepAI() {
  const a = E.hasLegal(state, AI) ? aiMove(state, aiLevel) : E.PASS;
  state = E.applyAction(state, a);
  if (state.current === AI && !E.isTerminal(state)) { setTimeout(stepAI, 250); return; }
  busy = false;
  afterMove();
}

function newGame() {
  state = E.initialState(HUMAN); sel = { pid: null, variant: null };
  legalRefs = new Set(); hoverIdx = null; busy = false;
  render();
}

// ---- wire up -------------------------------------------------------------
buildPanels();
$("rotate").addEventListener("click", () => reorient(E.ROT_OF));
$("flip").addEventListener("click", () => reorient(E.FLIP_OF));
$("skip").addEventListener("click", () => { sel = { pid: null, variant: null }; legalRefs = new Set(); render(); });
$("pass").addEventListener("click", () => { if (!$("pass").disabled) { state = E.applyAction(state, E.PASS); afterMove(); } });
$("newgame").addEventListener("click", newGame);
$("level").addEventListener("change", (e) => { aiLevel = +e.target.value; });
canvas.addEventListener("mousemove", (e) => { const i = pixelToIdx(e); if (i !== hoverIdx) { hoverIdx = i; drawBoard(); } });
canvas.addEventListener("mouseleave", () => { hoverIdx = null; drawBoard(); });
canvas.addEventListener("click", (e) => place(pixelToIdx(e)));
render();
