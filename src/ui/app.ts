// App wiring: mode UI (play vs bot / bot vs bot), the setInterval game loop,
// async bot seats (one inference request in flight per seat), the HUD, and a
// requestAnimationFrame render loop reading the pure GameSession.

import type { Action, Observation } from "../engine/types";
import type { WorkerResponse } from "../model/worker";
import { drawBoard, type BoardView } from "./board";
import { HumanInput } from "./input";
import { GameSession, PASS_ACTION, moveToAction } from "./session";
import "./style.css";

type Mode = "human" | "bots";
type ModelKey = "champion" | "spatial-v5";

const MODEL_FILES: Record<ModelKey, string> = {
  champion: "a100-i-draw2.onnx",
  "spatial-v5": "spatial-v5.onnx",
};

function modelUrl(key: ModelKey): string {
  return `${import.meta.env.BASE_URL}models/${MODEL_FILES[key]}`;
}

function randomSeed(): number {
  return Math.floor(Math.random() * 0x80000000);
}

const HUMAN_HINT =
  "Click one of your tiles, then queue moves with arrow keys / WASD — the cursor follows the queue. " +
  "Hold Z for a 50% split, Q clears the queue.";
const BOTS_HINT = "Spectating bot vs bot — full map, no fog. Pick a model per seat, pause and step through ticks.";

/**
 * One bot seat: a model worker plus the async one-request-in-flight protocol.
 * The app asks for an action when a tick opens; if the answer has not arrived
 * by the next tick, the seat passes.
 */
class BotSeat {
  status: "loading" | "ready" | "error" = "loading";
  errorMessage = "";
  private worker: Worker;
  private inflight = false;
  private pending: Action | null = null;

  constructor(
    readonly url: string,
    private readonly onReady: () => void,
  ) {
    this.worker = this.spawn();
  }

  private spawn(): Worker {
    const w = new Worker(new URL("../model/worker.ts", import.meta.url), { type: "module" });
    w.onmessage = (e: MessageEvent<WorkerResponse>) => this.handle(e.data);
    w.postMessage({ type: "init", modelUrl: this.url });
    return w;
  }

  private handle(msg: WorkerResponse): void {
    switch (msg.type) {
      case "ready":
        this.status = "ready";
        this.onReady();
        break;
      case "action":
        this.inflight = false;
        this.pending = msg.action;
        break;
      case "error":
        this.inflight = false;
        this.status = "error";
        this.errorMessage = msg.message;
        break;
    }
  }

  /**
   * Fresh LSTM carry for a new game. If a request is somehow still in flight,
   * rebuild the worker so a stale action can never leak into the new game.
   */
  reset(): void {
    this.pending = null;
    if (this.inflight) {
      this.worker.terminate();
      this.status = "loading";
      this.inflight = false;
      this.worker = this.spawn();
      return;
    }
    this.worker.postMessage({ type: "reset" });
  }

  /** Ask for an action on this observation; no-op unless idle and ready. */
  request(obs: Observation): void {
    if (this.status !== "ready" || this.inflight) return;
    this.inflight = true;
    this.worker.postMessage({ type: "act", obs });
  }

  /** Take the latest answered action, if any. */
  take(): Action | null {
    const a = this.pending;
    this.pending = null;
    return a;
  }

  dispose(): void {
    this.worker.terminate();
  }
}

function el<T extends HTMLElement>(root: HTMLElement, selector: string): T {
  const node = root.querySelector<T>(selector);
  if (!node) throw new Error(`missing element: ${selector}`);
  return node;
}

export function mountApp(root: HTMLElement): void {
  new App(root);
}

class App {
  private mode: Mode = "human";
  private session: GameSession = new GameSession(randomSeed());
  private input: HumanInput | null = null;
  private seats: [BotSeat | null, BotSeat | null] = [null, null];
  private modelKeys: [ModelKey, ModelKey] = ["champion", "spatial-v5"];
  private humanSpeed = 1;
  private botTps = 8;
  private running = true;
  private timer: number | null = null;
  private readonly hudCache = new Map<string, string>();

  private readonly canvas: HTMLCanvasElement;
  private readonly turnEl: HTMLElement;
  private readonly nameEls: [HTMLElement, HTMLElement];
  private readonly armyEls: [HTMLElement, HTMLElement];
  private readonly landEls: [HTMLElement, HTMLElement];
  private readonly statusEl: HTMLElement;
  private readonly hintEl: HTMLElement;
  private readonly overlayEl: HTMLElement;
  private readonly winnerEl: HTMLElement;
  private readonly winnerSubEl: HTMLElement;
  private readonly seedInput: HTMLInputElement;
  private readonly humanCtlEl: HTMLElement;
  private readonly botsCtlEl: HTMLElement;
  private readonly pauseBtn: HTMLButtonElement;
  private readonly stepBtn: HTMLButtonElement;
  private readonly tpsInput: HTMLInputElement;
  private readonly tpsLabel: HTMLElement;
  private readonly modelSelects: [HTMLSelectElement, HTMLSelectElement];

  constructor(root: HTMLElement) {
    root.innerHTML = `
      <div class="turnbox" id="turn">Turn 0</div>
      <table class="leaderboard" title="army / land">
        <thead><tr><th>Player</th><th>Army</th><th>Land</th></tr></thead>
        <tbody>
          <tr><td class="name p0" id="p0-name">you</td><td id="p0-army">1</td><td id="p0-land">1</td></tr>
          <tr><td class="name p1" id="p1-name">champion</td><td id="p1-army">1</td><td id="p1-land">1</td></tr>
        </tbody>
      </table>
      <header class="topbar">
        <select id="mode" title="Game mode">
          <option value="human">Play vs bot</option>
          <option value="bots">Bot vs bot</option>
        </select>
        <div class="ctl seg" id="human-ctl">
          <button data-speed="1" class="active">1x</button>
          <button data-speed="2">2x</button>
          <button data-speed="5">5x</button>
        </div>
        <div class="ctl" id="bots-ctl" hidden>
          <button id="pause">Pause</button>
          <button id="step" disabled>Step</button>
          <input id="tps" type="range" min="1" max="20" value="8" title="Ticks per second" />
          <span class="tps" id="tps-label">8 t/s</span>
          <select id="model0" title="Blue (P0) model">
            <option value="champion" selected>P0: champion</option>
            <option value="spatial-v5">P0: spatial-v5</option>
          </select>
          <select id="model1" title="Red (P1) model">
            <option value="champion">P1: champion</option>
            <option value="spatial-v5" selected>P1: spatial-v5</option>
          </select>
        </div>
        <div class="ctl">
          <input id="seed" type="number" min="0" step="1" title="Map seed" />
          <button id="rand" title="Random seed">🎲</button>
          <button id="newgame" class="primary">New game</button>
        </div>
      </header>
      <main class="stage">
        <div class="board-wrap">
          <canvas id="board"></canvas>
          <div class="overlay" id="overlay" hidden>
            <div class="card">
              <div class="winner" id="winner-text"></div>
              <div class="sub" id="winner-sub"></div>
              <button id="rematch" class="primary">Rematch</button>
            </div>
          </div>
        </div>
        <div class="hint" id="hint"></div>
        <div class="status" id="status"></div>
      </main>`;

    this.canvas = el(root, "#board");
    this.turnEl = el(root, "#turn");
    this.nameEls = [el(root, "#p0-name"), el(root, "#p1-name")];
    this.armyEls = [el(root, "#p0-army"), el(root, "#p1-army")];
    this.landEls = [el(root, "#p0-land"), el(root, "#p1-land")];
    this.statusEl = el(root, "#status");
    this.hintEl = el(root, "#hint");
    this.overlayEl = el(root, "#overlay");
    this.winnerEl = el(root, "#winner-text");
    this.winnerSubEl = el(root, "#winner-sub");
    this.seedInput = el(root, "#seed");
    this.humanCtlEl = el(root, "#human-ctl");
    this.botsCtlEl = el(root, "#bots-ctl");
    this.pauseBtn = el(root, "#pause");
    this.stepBtn = el(root, "#step");
    this.tpsInput = el(root, "#tps");
    this.tpsLabel = el(root, "#tps-label");
    this.modelSelects = [el(root, "#model0"), el(root, "#model1")];

    el<HTMLSelectElement>(root, "#mode").addEventListener("change", (e) => {
      this.setMode((e.target as HTMLSelectElement).value as Mode);
    });
    root.querySelectorAll<HTMLButtonElement>("#human-ctl button").forEach((b) => {
      b.addEventListener("click", () => this.setHumanSpeed(Number(b.dataset.speed)));
    });
    this.pauseBtn.addEventListener("click", () => this.togglePause());
    this.stepBtn.addEventListener("click", () => {
      if (!this.running) this.tick();
    });
    this.tpsInput.addEventListener("input", () => this.setBotTps(Number(this.tpsInput.value)));
    this.modelSelects[0].addEventListener("change", () =>
      this.setModel(0, this.modelSelects[0].value as ModelKey),
    );
    this.modelSelects[1].addEventListener("change", () =>
      this.setModel(1, this.modelSelects[1].value as ModelKey),
    );
    el<HTMLButtonElement>(root, "#newgame").addEventListener("click", () =>
      this.newGame(this.parseSeed()),
    );
    el<HTMLButtonElement>(root, "#rand").addEventListener("click", () => {
      this.seedInput.value = String(randomSeed());
      this.newGame(this.parseSeed());
    });
    el<HTMLButtonElement>(root, "#rematch").addEventListener("click", () =>
      this.newGame(randomSeed()),
    );
    this.seedInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") this.newGame(this.parseSeed());
    });

    // Preload the display font so canvas army counts render with it.
    document.fonts?.load("700 15px Quicksand").catch(() => {});

    this.applyModeUi();
    this.newGame(this.session.seed);
    requestAnimationFrame(this.frame);
    // Debug/testing handle (headless smoke tests read game state through it).
    (window as unknown as { __app: App }).__app = this;
  }

  // ---- mode + controls ----------------------------------------------------

  private setMode(mode: Mode): void {
    if (mode === this.mode) return;
    this.mode = mode;
    this.applyModeUi();
    this.newGame(this.parseSeed());
  }

  private applyModeUi(): void {
    const human = this.mode === "human";
    this.humanCtlEl.hidden = !human;
    this.botsCtlEl.hidden = human;
    this.hintEl.textContent = human ? HUMAN_HINT : BOTS_HINT;
  }

  private setHumanSpeed(speed: number): void {
    if (![1, 2, 5].includes(speed)) return;
    this.humanSpeed = speed;
    this.humanCtlEl.querySelectorAll("button").forEach((b) => {
      b.classList.toggle("active", Number((b as HTMLButtonElement).dataset.speed) === speed);
    });
    this.startLoop();
  }

  private setBotTps(tps: number): void {
    this.botTps = Math.min(20, Math.max(1, Math.round(tps)));
    this.tpsLabel.textContent = `${this.botTps} t/s`;
    this.startLoop();
  }

  private setModel(seat: 0 | 1, key: ModelKey): void {
    this.modelKeys[seat] = key;
    this.newGame(this.parseSeed());
  }

  private togglePause(): void {
    this.running = !this.running;
    this.updatePauseUi();
    this.startLoop();
  }

  private updatePauseUi(): void {
    this.pauseBtn.textContent = this.running ? "Pause" : "Resume";
    this.stepBtn.disabled = this.running;
  }

  private parseSeed(): number {
    const n = Number.parseInt(this.seedInput.value, 10);
    return Number.isFinite(n) ? n >>> 0 : randomSeed();
  }

  // ---- game lifecycle -----------------------------------------------------

  private newGame(seed: number): void {
    this.stopLoop();
    this.session = new GameSession(seed);
    this.seedInput.value = String(seed);
    this.overlayEl.hidden = true;

    this.input?.detach();
    this.input = null;
    if (this.mode === "human") {
      this.releaseSeat(0);
      this.ensureSeat(1, "champion");
      this.input = new HumanInput(this.session, this.canvas);
      this.input.attach();
    } else {
      this.ensureSeat(0, this.modelKeys[0]);
      this.ensureSeat(1, this.modelKeys[1]);
    }
    this.seats[0]?.reset();
    this.seats[1]?.reset();
    this.requestBots();

    this.running = true;
    this.updatePauseUi();
    this.startLoop();
  }

  private finish(): void {
    this.stopLoop();
    const w = this.session.winner;
    const human = this.mode === "human";
    this.winnerEl.textContent = human
      ? w === 0
        ? "You win!"
        : "The bot wins."
      : w === 0
        ? "Blue wins!"
        : "Red wins!";
    this.winnerEl.className = `winner p${w}`;
    this.winnerSubEl.textContent = `after ${this.session.turn} turns`;
    this.overlayEl.hidden = false;
  }

  // ---- bot seats ----------------------------------------------------------

  private ensureSeat(seat: 0 | 1, key: ModelKey): void {
    const url = modelUrl(key);
    const existing = this.seats[seat];
    if (existing && existing.url === url) return;
    existing?.dispose();
    this.seats[seat] = new BotSeat(url, () => this.onSeatReady(seat));
  }

  private releaseSeat(seat: 0 | 1): void {
    this.seats[seat]?.dispose();
    this.seats[seat] = null;
  }

  private onSeatReady(seat: 0 | 1): void {
    if (this.session.done) return;
    if (this.mode === "human" && seat !== 1) return;
    this.seats[seat]?.request(this.session.obs(seat));
  }

  private requestBots(): void {
    if (this.session.done) return;
    if (this.mode === "human") {
      this.seats[1]?.request(this.session.obs(1));
    } else {
      this.seats[0]?.request(this.session.obs(0));
      this.seats[1]?.request(this.session.obs(1));
    }
  }

  // ---- game loop ----------------------------------------------------------

  private tickMs(): number {
    return this.mode === "human" ? 500 / this.humanSpeed : 1000 / this.botTps;
  }

  private startLoop(): void {
    this.stopLoop();
    if (!this.running || this.session.done) return;
    this.timer = window.setInterval(() => this.tick(), this.tickMs());
  }

  private stopLoop(): void {
    if (this.timer !== null) {
      clearInterval(this.timer);
      this.timer = null;
    }
  }

  private tick(): void {
    if (this.session.done) {
      this.finish();
      return;
    }
    let actions: [Action, Action];
    if (this.mode === "human") {
      const m = this.session.dequeue(0);
      actions = [m ? moveToAction(m) : PASS_ACTION, this.seats[1]?.take() ?? PASS_ACTION];
    } else {
      actions = [
        this.seats[0]?.take() ?? PASS_ACTION,
        this.seats[1]?.take() ?? PASS_ACTION,
      ];
    }
    this.session.step(actions);
    if (this.session.done) {
      this.finish();
      return;
    }
    this.requestBots();
  }

  // ---- rendering + HUD ----------------------------------------------------

  private readonly frame = (now: number): void => {
    this.draw(now);
    requestAnimationFrame(this.frame);
  };

  private setText(elm: HTMLElement, key: string, text: string): void {
    if (this.hudCache.get(key) === text) return;
    this.hudCache.set(key, text);
    elm.textContent = text;
  }

  private draw(now: number): void {
    const human = this.mode === "human";
    // Human mode renders strictly through the fogged observation.
    const obs = human ? this.session.obs(0) : null;
    const view: BoardView = obs ? { kind: "obs", obs } : { kind: "state", state: this.session.state };
    drawBoard(
      this.canvas,
      view,
      {
        selected: human ? (this.input?.anchor ?? null) : null,
        queue: human ? this.session.queueOf(0) : [],
      },
      now,
    );

    this.setText(this.turnEl, "turn", `Turn ${this.session.turn}`);
    const names: [string, string] = human
      ? ["you", "champion"]
      : [this.modelKeys[0], this.modelKeys[1]];
    this.setText(this.nameEls[0], "n0", names[0]);
    this.setText(this.nameEls[1], "n1", names[1]);
    let army: [number, number];
    let land: [number, number];
    if (obs) {
      army = [obs.ownedArmyCount, obs.opponentArmyCount];
      land = [obs.ownedLandCount, obs.opponentLandCount];
    } else {
      const t = this.session.totals();
      army = t.army;
      land = t.land;
    }
    this.setText(this.armyEls[0], "a0", String(army[0]));
    this.setText(this.armyEls[1], "a1", String(army[1]));
    this.setText(this.landEls[0], "l0", String(land[0]));
    this.setText(this.landEls[1], "l1", String(land[1]));

    const status = human
      ? `bot · ${this.seatStatus(1)}`
      : `P0 · ${this.seatStatus(0)}  |  P1 · ${this.seatStatus(1)}`;
    this.setText(this.statusEl, "status", status);
    this.statusEl.classList.toggle(
      "error",
      this.seats.some((s) => s?.status === "error"),
    );
  }

  private seatStatus(seat: 0 | 1): string {
    const s = this.seats[seat];
    if (!s) return "—";
    const name = this.mode === "human" ? "champion" : this.modelKeys[seat];
    if (s.status === "loading") return `${name} loading…`;
    if (s.status === "error") return `${name} failed (${s.errorMessage})`;
    return `${name} ready`;
  }
}
