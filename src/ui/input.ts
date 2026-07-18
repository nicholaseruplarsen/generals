// Human input, generals.io-style: click an owned tile to select it; arrow
// keys / WASD enqueue a move from the anchor, and the anchor follows the
// queued destination. Holding Z marks a 50% split; Q clears the queue.

import { DIRECTIONS, H, W } from "../engine/types";
import type { GameSession } from "./session";

const KEY_DIRS: Record<string, number> = {
  arrowup: 0,
  w: 0,
  arrowdown: 1,
  s: 1,
  arrowleft: 2,
  a: 2,
  arrowright: 3,
  d: 3,
};

export class HumanInput {
  /** Flat index of the selection anchor, or null when nothing is selected. */
  anchor: number | null = null;
  private zDown = false;
  private attached = false;

  constructor(
    private readonly session: GameSession,
    private readonly canvas: HTMLCanvasElement,
    private readonly player: 0 | 1 = 0,
  ) {}

  attach(): void {
    if (this.attached) return;
    this.attached = true;
    this.canvas.addEventListener("pointerdown", this.onPointerDown);
    window.addEventListener("keydown", this.onKeyDown);
    window.addEventListener("keyup", this.onKeyUp);
    window.addEventListener("blur", this.onBlur);
  }

  detach(): void {
    if (!this.attached) return;
    this.attached = false;
    this.canvas.removeEventListener("pointerdown", this.onPointerDown);
    window.removeEventListener("keydown", this.onKeyDown);
    window.removeEventListener("keyup", this.onKeyUp);
    window.removeEventListener("blur", this.onBlur);
    this.reset();
  }

  reset(): void {
    this.anchor = null;
    this.zDown = false;
  }

  private onPointerDown = (e: PointerEvent): void => {
    // Clicking the board always reclaims the keyboard — if a form control
    // (seed box, selects) kept focus, arrows would go to it, not the game.
    if (document.activeElement instanceof HTMLElement) document.activeElement.blur();
    const rect = this.canvas.getBoundingClientRect();
    const col = Math.floor(((e.clientX - rect.left) / rect.width) * W);
    const row = Math.floor(((e.clientY - rect.top) / rect.height) * H);
    if (row < 0 || row >= H || col < 0 || col >= W) return;
    const idx = row * W + col;
    // Clicking a tile you own selects / re-anchors; anything else is ignored.
    if (this.session.obs(this.player).ownedCells[idx] === 1) this.anchor = idx;
  };

  private onKeyDown = (e: KeyboardEvent): void => {
    const t = e.target;
    if (
      t instanceof HTMLInputElement ||
      t instanceof HTMLSelectElement ||
      t instanceof HTMLTextAreaElement
    ) {
      return;
    }
    const key = e.key.toLowerCase();
    if (key === "z") {
      this.zDown = true;
      return;
    }
    if (key === "q") {
      this.session.clearQueue(this.player);
      return;
    }
    const dir = KEY_DIRS[key];
    if (dir === undefined) return;
    e.preventDefault(); // keep arrow keys from scrolling the page
    this.queueMove(dir);
  };

  private onKeyUp = (e: KeyboardEvent): void => {
    if (e.key.toLowerCase() === "z") this.zDown = false;
  };

  private onBlur = (): void => {
    this.zDown = false;
  };

  private queueMove(dir: number): void {
    if (this.anchor === null || this.session.done) return;
    const row = Math.floor(this.anchor / W);
    const col = this.anchor % W;
    const d = DIRECTIONS[dir]!;
    const nr = row + d[0];
    const nc = col + d[1];
    if (nr < 0 || nr >= H || nc < 0 || nc >= W) return;
    // Same blocked rule as the policy's action mask: never queue into a
    // visible mountain or a fogged structure — otherwise the anchor follows
    // the queue onto an impassable cell and every later keypress is dead.
    const dest = nr * W + nc;
    const obs = this.session.obs(this.player);
    if (obs.mountains[dest] === 1 || obs.structuresInFog[dest] === 1) return;
    this.session.enqueue(this.player, { row, col, dir, split: this.zDown });
    this.anchor = dest; // selection follows the queued destination
  }
}
