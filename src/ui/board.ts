// Canvas board renderer — generals.io look: ~48px tiles, player colours, fog.
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

const COLOR_BG = "#0b0e13";
const COLOR_NEUTRAL = "#4b5563";
const COLOR_PLAYER: readonly [string, string] = ["#4363d8", "#e6194b"];
const COLOR_MOUNTAIN = "#242a33";
const COLOR_MOUNTAIN_GLYPH = "#7d8794";
const COLOR_FOG = "#151a21";
const COLOR_FOG_GLYPH = "#3d4654";
const COLOR_CITY = "rgba(255, 255, 255, 0.92)";
const COLOR_CROWN = "#ffd54a";
const COLOR_TEXT = "#ffffff";
const COLOR_ARROW = "rgba(255, 255, 255, 0.95)";
const COLOR_ARROW_SPLIT = "#ffd166";

interface CellInfo {
  fill: string;
  /** Mountain or city hidden in fog — both drawn with the identical glyph. */
  fogObstacle: boolean;
  mountain: boolean;
  city: boolean;
  general: boolean;
  army: number;
}

function fillFor(owner: -1 | 0 | 1): string {
  return owner === -1 ? COLOR_NEUTRAL : COLOR_PLAYER[owner];
}

const EMPTY_CELL: CellInfo = {
  fill: COLOR_FOG,
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
    return { ...EMPTY_CELL, fill: COLOR_MOUNTAIN, mountain: true };
  }
  const owner = obs.ownedCells[i] === 1 ? 0 : obs.opponentCells[i] === 1 ? 1 : -1;
  return {
    fill: fillFor(owner),
    fogObstacle: false,
    mountain: false,
    city: obs.cities[i] === 1,
    general: obs.generals[i] === 1,
    army: obs.armies[i]!,
  };
}

function cellFromState(s: GameState, i: number): CellInfo {
  if (s.mountains[i] === 1) {
    return { ...EMPTY_CELL, fill: COLOR_MOUNTAIN, mountain: true };
  }
  const owner = s.owner0[i] === 1 ? 0 : s.owner1[i] === 1 ? 1 : -1;
  return {
    fill: fillFor(owner),
    fogObstacle: false,
    mountain: false,
    city: s.cities[i] === 1,
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

function drawTriangle(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  size: number,
  color: string,
): void {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(cx, cy - size);
  ctx.lineTo(cx - size * 0.95, cy + size * 0.8);
  ctx.lineTo(cx + size * 0.95, cy + size * 0.8);
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
  ctx.shadowColor = "rgba(0, 0, 0, 0.55)";
  ctx.shadowBlur = 3;
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
    ctx.font = "700 11px system-ui, sans-serif";
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
  ctx.fillStyle = COLOR_BG;
  ctx.fillRect(0, 0, W * TILE, H * TILE);

  const cells: CellInfo[] = new Array<CellInfo>(H * W);
  for (let r = 0; r < H; r++) {
    for (let c = 0; c < W; c++) {
      const i = r * W + c;
      const cell = view.kind === "obs" ? cellFromObs(view.obs, i) : cellFromState(view.state, i);
      cells[i] = cell;
      ctx.fillStyle = cell.fill;
      ctx.fillRect(c * TILE + 1, r * TILE + 1, TILE - 2, TILE - 2);
      if (cell.fogObstacle || cell.mountain) {
        drawTriangle(
          ctx,
          c * TILE + TILE / 2,
          r * TILE + TILE / 2,
          9,
          cell.fogObstacle ? COLOR_FOG_GLYPH : COLOR_MOUNTAIN_GLYPH,
        );
      }
    }
  }

  // Queued-move arrows, under the army counts so text stays readable. The
  // next move to execute (queue front) is drawn brighter.
  overlay.queue.forEach((m, order) => drawArrow(ctx, m, order === 0 ? 1 : 0.55));

  // Glyphs + army counts.
  for (let r = 0; r < H; r++) {
    for (let c = 0; c < W; c++) {
      const cell = cells[r * W + c]!;
      if (cell.mountain || cell.fogObstacle) continue;
      const cx = c * TILE + TILE / 2;
      const hasGlyph = cell.city || cell.general;
      if (cell.city) {
        ctx.fillStyle = COLOR_CITY;
        ctx.beginPath();
        ctx.arc(cx, r * TILE + 15, 7, 0, Math.PI * 2);
        ctx.fill();
      }
      if (cell.general) {
        drawText(ctx, "♛", cx, r * TILE + 14, "16px 'Segoe UI Symbol', serif", COLOR_CROWN);
      }
      if (cell.army > 0) {
        const cy = hasGlyph ? r * TILE + TILE * 0.7 : r * TILE + TILE / 2;
        drawText(ctx, String(cell.army), cx, cy, "700 16px system-ui, sans-serif", COLOR_TEXT);
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
    ctx.strokeRect(sc * TILE + 2.5, sr * TILE + 2.5, TILE - 5, TILE - 5);
  }
}
