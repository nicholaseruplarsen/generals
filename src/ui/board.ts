// Canvas board renderer, styled after official generals.io: light tiles on a
// dark fog panel, azure/red players, drawn mountain/city/crown glyphs.
// Human mode renders a fogged Observation (never the true state); bot-vs-bot
// spectator mode renders the full GameState.

import { DIRECTIONS, H, W } from "../engine/types";
import type { GameState, Observation } from "../engine/types";
import type { QueuedMove } from "./session";

export const TILE = 48;

export type BoardView =
  | { kind: "obs"; obs: Observation }
  | { kind: "state"; state: GameState };

export interface BoardOverlay {
  /** Flat index of the selection anchor, or null. */
  selected: number | null;
  /** Pending queued moves; front of the queue is next to execute. */
  queue: readonly QueuedMove[];
}

// Palette sampled from official generals.io.
const COLOR_EMPTY = "#dcdcdc"; //   visible unowned ground
const COLOR_NEUTRAL_CITY = "#949494";
const COLOR_MOUNTAIN_TILE = "#bbbbbb";
const COLOR_PLAYER: readonly [string, string] = ["#4a87f0", "#ee342f"];
const COLOR_FOG = "#3a3a3a"; //     fog ground = the dark panel itself
const COLOR_GLYPH = "#1c1c1c"; //   glyphs on visible tiles
const COLOR_GLYPH_FOG = "#141414"; // obstacle glyphs inside fog
const COLOR_GRID_LIGHT = "rgba(0, 0, 0, 0.28)";
const COLOR_GRID_FOG = "rgba(0, 0, 0, 0.18)";
const COLOR_TEXT = "#ffffff";
const COLOR_ARROW = "rgba(255, 255, 255, 0.95)";
const COLOR_ARROW_SPLIT = "#ffd166";

const FONT = "Quicksand, 'Trebuchet MS', system-ui, sans-serif";

interface CellInfo {
  fill: string;
  visible: boolean;
  /** Mountain or city hidden in fog — both drawn with the identical glyph. */
  fogObstacle: boolean;
  mountain: boolean;
  city: boolean;
  general: boolean;
  army: number;
}

const EMPTY_CELL: CellInfo = {
  fill: COLOR_FOG,
  visible: false,
  fogObstacle: false,
  mountain: false,
  city: false,
  general: false,
  army: 0,
};

function cellFromObs(obs: Observation, i: number): CellInfo {
  if (obs.fogCells[i] === 1) return EMPTY_CELL;
  if (obs.structuresInFog[i] === 1) return { ...EMPTY_CELL, fogObstacle: true };
  if (obs.mountains[i] === 1) {
    return { ...EMPTY_CELL, visible: true, fill: COLOR_MOUNTAIN_TILE, mountain: true };
  }
  const owner = obs.ownedCells[i] === 1 ? 0 : obs.opponentCells[i] === 1 ? 1 : -1;
  const city = obs.cities[i] === 1;
  return {
    fill: owner === -1 ? (city ? COLOR_NEUTRAL_CITY : COLOR_EMPTY) : COLOR_PLAYER[owner],
    visible: true,
    fogObstacle: false,
    mountain: false,
    city,
    general: obs.generals[i] === 1,
    army: obs.armies[i]!,
  };
}

function cellFromState(s: GameState, i: number): CellInfo {
  if (s.mountains[i] === 1) {
    return { ...EMPTY_CELL, visible: true, fill: COLOR_MOUNTAIN_TILE, mountain: true };
  }
  const owner = s.owner0[i] === 1 ? 0 : s.owner1[i] === 1 ? 1 : -1;
  const city = s.cities[i] === 1;
  return {
    fill: owner === -1 ? (city ? COLOR_NEUTRAL_CITY : COLOR_EMPTY) : COLOR_PLAYER[owner],
    visible: true,
    fogObstacle: false,
    mountain: false,
    city,
    general: s.generals[i] === 1,
    army: s.armies[i]!,
  };
}

const dprCache = new WeakMap<HTMLCanvasElement, number>();

function ctx2d(canvas: HTMLCanvasElement): CanvasRenderingContext2D {
  const dpr = window.devicePixelRatio || 1;
  if (dprCache.get(canvas) !== dpr) {
    canvas.width = W * TILE * dpr;
    canvas.height = H * TILE * dpr;
    dprCache.set(canvas, dpr);
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("2d canvas context unavailable");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

/** Official-style double-peak mountain (stroked polyline). */
function drawMountain(ctx: CanvasRenderingContext2D, cx: number, cy: number, color: string): void {
  const s = TILE * 0.42;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2.5;
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(cx - s * 0.95, cy + s * 0.55);
  ctx.lineTo(cx - s * 0.35, cy - s * 0.5);
  ctx.lineTo(cx + s * 0.05, cy + s * 0.05);
  ctx.lineTo(cx + s * 0.35, cy - s * 0.25);
  ctx.lineTo(cx + s * 0.95, cy + s * 0.55);
  ctx.stroke();
}

/** Little castle tower, like the official city icon. */
function drawCity(ctx: CanvasRenderingContext2D, cx: number, cy: number, color: string): void {
  const w = TILE * 0.42;
  const h = TILE * 0.4;
  const x = cx - w / 2;
  const y = cy - h / 2;
  ctx.fillStyle = color;
  // battlement: three merlons
  const mw = w / 5;
  ctx.fillRect(x, y, mw, h * 0.3);
  ctx.fillRect(x + 2 * mw, y, mw, h * 0.3);
  ctx.fillRect(x + 4 * mw, y, mw, h * 0.3);
  // body + door notch
  ctx.fillRect(x, y + h * 0.22, w, h * 0.78);
  ctx.clearRect(cx - w * 0.12, y + h * 0.62, w * 0.24, h * 0.38);
}

/** Three-spike crown for generals. */
function drawCrown(ctx: CanvasRenderingContext2D, cx: number, cy: number, color: string): void {
  const w = TILE * 0.62;
  const h = TILE * 0.48;
  const x = cx - w / 2;
  const base = cy + h / 2;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(x, base);
  ctx.lineTo(x, base - h * 0.55);
  ctx.lineTo(x + w * 0.25, base - h * 0.3);
  ctx.lineTo(x + w * 0.5, base - h);
  ctx.lineTo(x + w * 0.75, base - h * 0.3);
  ctx.lineTo(x + w, base - h * 0.55);
  ctx.lineTo(x + w, base);
  ctx.closePath();
  ctx.fill();
}

function drawText(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  font: string,
  color: string,
): void {
  ctx.font = font;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = color;
  ctx.shadowColor = "rgba(0, 0, 0, 0.75)";
  ctx.shadowBlur = 2;
  ctx.shadowOffsetY = 1;
  ctx.fillText(text, x, y);
  ctx.shadowColor = "transparent";
  ctx.shadowBlur = 0;
  ctx.shadowOffsetY = 0;
}

function drawArrow(ctx: CanvasRenderingContext2D, m: QueuedMove, alpha: number): void {
  const d = DIRECTIONS[m.dir]!;
  const x0 = (m.col + 0.5) * TILE;
  const y0 = (m.row + 0.5) * TILE;
  const x1 = x0 + d[1] * TILE;
  const y1 = y0 + d[0] * TILE;
  const sx = x0 + d[1] * TILE * 0.2;
  const sy = y0 + d[0] * TILE * 0.2;
  const ex = x1 - d[1] * TILE * 0.3;
  const ey = y1 - d[0] * TILE * 0.3;
  const color = m.split ? COLOR_ARROW_SPLIT : COLOR_ARROW;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = 5;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(sx, sy);
  ctx.lineTo(ex, ey);
  ctx.stroke();
  const angle = Math.atan2(ey - sy, ex - sx);
  for (const off of [2.6, -2.6]) {
    ctx.beginPath();
    ctx.moveTo(ex, ey);
    ctx.lineTo(ex + Math.cos(angle + off) * 11, ey + Math.sin(angle + off) * 11);
    ctx.stroke();
  }
  if (m.split) {
    // ½ marker just off the shaft midpoint.
    const mx = (sx + ex) / 2 - (ey - sy) * 0.35;
    const my = (sy + ey) / 2 + (ex - sx) * 0.35;
    ctx.fillStyle = color;
    ctx.font = `700 11px ${FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("½", mx, my);
  }
  ctx.restore();
}

export function drawBoard(
  canvas: HTMLCanvasElement,
  view: BoardView,
  overlay: BoardOverlay,
  now: number,
): void {
  const ctx = ctx2d(canvas);
  ctx.fillStyle = COLOR_FOG;
  ctx.fillRect(0, 0, W * TILE, H * TILE);

  const cells: CellInfo[] = new Array<CellInfo>(H * W);
  for (let r = 0; r < H; r++) {
    for (let c = 0; c < W; c++) {
      const i = r * W + c;
      const cell = view.kind === "obs" ? cellFromObs(view.obs, i) : cellFromState(view.state, i);
      cells[i] = cell;
      if (cell.fill !== COLOR_FOG) {
        ctx.fillStyle = cell.fill;
        ctx.fillRect(c * TILE, r * TILE, TILE, TILE);
      }
      // thin grid, lighter inside fog like the official board
      ctx.strokeStyle = cell.visible ? COLOR_GRID_LIGHT : COLOR_GRID_FOG;
      ctx.lineWidth = 1;
      ctx.strokeRect(c * TILE + 0.5, r * TILE + 0.5, TILE - 1, TILE - 1);

      const cx = c * TILE + TILE / 2;
      const cy = r * TILE + TILE / 2;
      if (cell.fogObstacle) drawMountain(ctx, cx, cy, COLOR_GLYPH_FOG);
      else if (cell.mountain) drawMountain(ctx, cx, cy, COLOR_GLYPH);
    }
  }

  // Queued-move arrows, under the army counts so text stays readable. The
  // next move to execute (queue front) is drawn brighter.
  overlay.queue.forEach((m, order) => drawArrow(ctx, m, order === 0 ? 1 : 0.55));

  // Structure glyphs + army counts.
  for (let r = 0; r < H; r++) {
    for (let c = 0; c < W; c++) {
      const cell = cells[r * W + c]!;
      if (cell.mountain || cell.fogObstacle) continue;
      const cx = c * TILE + TILE / 2;
      const cy = r * TILE + TILE / 2;
      if (cell.city) drawCity(ctx, cx, cy, COLOR_GLYPH);
      if (cell.general) drawCrown(ctx, cx, cy, "rgba(0, 0, 0, 0.55)");
      if (cell.army > 0) {
        drawText(ctx, String(cell.army), cx, cy, `700 15px ${FONT}`, COLOR_TEXT);
      }
    }
  }

  // Pulsing selection anchor.
  if (overlay.selected !== null) {
    const sr = Math.floor(overlay.selected / W);
    const sc = overlay.selected % W;
    const pulse = 0.7 + 0.3 * Math.sin(now / 160);
    ctx.strokeStyle = `rgba(255, 255, 255, ${pulse.toFixed(3)})`;
    ctx.lineWidth = 3;
    ctx.strokeRect(sc * TILE + 1.5, sr * TILE + 1.5, TILE - 3, TILE - 3);
  }
}
