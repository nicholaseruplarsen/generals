import { buildCost, createState, playerView, step, totals, visibility } from "./game";
import { competitionMapAt, randomCompetitionMap } from "./map";
import { PolicyMemory, preparePolicyInput } from "./observation";
import { SELFPLAY_REPLAYS, type SelfplayReplay } from "./selfplay.generated";
import {
  DIRECTIONS,
  PASS_ACTION,
  PAD_TO,
  type Action,
  type CompetitionState,
  type Player,
  type PolicyDecision,
} from "./types";
import type { CompetitionWorkerResponse } from "./worker";
import "./style.css";

type Mode = "human" | "selfplay";
const EMPTY_LOGITS = new Float32Array(4410);

function select<T extends HTMLElement>(root: HTMLElement, selector: string): T {
  /** Return one required UI element or fail at startup. */
  const element = root.querySelector<T>(selector);
  if (element === null) throw new Error(`Missing competition UI element ${selector}`);
  return element;
}

function modelUrl(): string {
  /** Resolve the static g08 ONNX artifact under Vite's configured base path. */
  return `${import.meta.env.BASE_URL}models/g08-champion.onnx`;
}

class ChampionSeat {
  /** One browser worker and one independent obs-v2 memory state. */
  readonly memory = new PolicyMemory();
  status: "loading" | "ready" | "error" = "loading";
  error = "";
  private readonly worker: Worker;
  private pending: ((decision: PolicyDecision) => void) | null = null;

  constructor(private readonly onStatus: () => void) {
    this.worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = (event: MessageEvent<CompetitionWorkerResponse>) => this.receive(event.data);
    this.worker.postMessage({ type: "init", modelUrl: modelUrl() });
  }

  private receive(message: CompetitionWorkerResponse): void {
    /** Resolve model lifecycle and exactly one in-flight action request. */
    if (message.type === "ready") {
      this.status = "ready";
      this.onStatus();
      return;
    }
    if (message.type === "error") {
      this.status = "error";
      this.error = message.message;
      const resolve = this.pending;
      this.pending = null;
      if (resolve !== null) {
        resolve({ action: PASS_ACTION, actionIndex: 8 * PAD_TO * PAD_TO, logits: new Float32Array(4410) });
      }
      this.onStatus();
      return;
    }
    const resolve = this.pending;
    this.pending = null;
    if (resolve !== null) resolve(message);
  }

  async act(state: CompetitionState, player: Player): Promise<PolicyDecision> {
    /** Evaluate g08 on one legal player view and advance that seat's memory. */
    if (this.status !== "ready" || this.pending !== null) {
      return { action: PASS_ACTION, actionIndex: 8 * PAD_TO * PAD_TO, logits: new Float32Array(4410) };
    }
    const input = preparePolicyInput(playerView(state, player), this.memory);
    return new Promise<PolicyDecision>((resolve) => {
      this.pending = resolve;
      this.worker.postMessage({ type: "act", input });
    });
  }

  dispose(): void {
    /** Terminate this seat's model worker. */
    this.worker.terminate();
  }
}

export function mountCompetitionApp(root: HTMLElement): void {
  /** Mount the competition game and start its render/update loops. */
  new CompetitionApp(root);
}

class CompetitionApp {
  private mode: Mode;
  private state = createState(randomCompetitionMap(), false);
  private opponentSeat: ChampionSeat | null = null;
  private selfplayReplay: SelfplayReplay = SELFPLAY_REPLAYS[0]!;
  private selfplayCursor = 0;
  private running = true;
  private busy = false;
  private tps = 2;
  private selected: number | null = this.state.generals[0];
  private half = false;
  private readonly humanQueue: Action[] = [];
  private lastDecision: readonly [PolicyDecision | null, PolicyDecision | null] = [null, null];
  private policyOverlay = false;
  private timer: number | null = null;
  private readonly humanSeen = new Uint8Array(PAD_TO * PAD_TO);
  private readonly humanMountains = new Uint8Array(PAD_TO * PAD_TO);
  private readonly humanCastles = new Uint8Array(PAD_TO * PAD_TO);

  private readonly board: HTMLElement;
  private readonly tiles: HTMLElement[] = [];
  private readonly tileNumbers: HTMLElement[] = [];
  private readonly tileArrows: HTMLElement[] = [];
  private readonly modeButtons: NodeListOf<HTMLButtonElement>;
  private readonly pauseButton: HTMLButtonElement;
  private readonly stepButton: HTMLButtonElement;
  private readonly status: HTMLElement;
  private readonly turn: HTMLElement;
  private readonly names: readonly [HTMLElement, HTMLElement];
  private readonly army: readonly [HTMLElement, HTMLElement];
  private readonly land: readonly [HTMLElement, HTMLElement];
  private readonly buildButton: HTMLButtonElement;
  private readonly halfButton: HTMLButtonElement;
  private readonly overlayButton: HTMLButtonElement;
  private readonly endScreen: HTMLElement;
  private readonly scoreRows: readonly [HTMLElement, HTMLElement];
  private readonly transportFeedback: HTMLElement;

  constructor(private readonly root: HTMLElement) {
    const params = new URLSearchParams(location.search);
    root.classList.toggle("is-embedded", params.get("embed") === "1");
    this.mode = params.get("mode") === "bots" ? "selfplay" : "human";
    if (this.mode === "selfplay") this.loadSelfplayReplay();
    root.innerHTML = `
      <div class="competition-shell">
        <header class="competition-header">
          <a class="competition-mark" href="${import.meta.env.BASE_URL.replace(/play\/$/, "")}">GENERALS<span>.COMPETITION</span></a>
          <nav><button class="mode-button" data-mode="human">PLAY G08</button><button class="mode-button" data-mode="selfplay">G08 SELF-PLAY</button></nav>
          <div class="model-badge"><i></i><span data-status>loading 2.72M parameters…</span></div>
        </header>
        <main class="competition-main pp-viewer">
          <div class="replay-viewer-meta"><span data-match-meta>YOU vs G08 · live competition sandbox</span><span>Tick <b data-turn>0</b> / 1200</span></div>
          <div class="replay-score">
            <div class="rs-row rs-blue is-leader" data-score-row-0><span class="rs-name" data-name-0>YOU</span><span class="rs-stat"><b data-land-0>1</b> land</span><span class="rs-stat"><b data-army-0>1</b> army</span></div>
            <div class="rs-row rs-red" data-score-row-1><span class="rs-name" data-name-1>G08</span><span class="rs-stat"><b data-land-1>1</b> land</span><span class="rs-stat"><b data-army-1>1</b> army</span></div>
          </div>
          <div class="replay-viewer-stage">
            <div class="replay-board" data-board aria-label="Interactive Generals competition board"></div>
            <div class="transport-feedback" data-transport-feedback aria-hidden="true"><span>❚❚</span></div>
            <div class="end-screen" data-end hidden><strong></strong><span></span><button type="button" data-rematch>REMATCH</button></div>
          </div>
          <div class="replay-controls">
            <button type="button" data-new title="Restart">◀◀</button>
            <button type="button" data-pause class="ctl-play" title="Pause">❚❚</button>
            <button type="button" data-step title="Advance one turn">▶</button>
            <input class="rscrub" type="range" min="0" max="1200" value="0" data-progress disabled aria-label="Current turn">
            <select class="ctl-speed" data-speed aria-label="Game speed"><option value="1">1×</option><option value="2" selected>2×</option><option value="4">4×</option><option value="8">8×</option></select>
          </div>
          <div class="replay-hint"><span><kbd>Click</kbd> source + destination</span><span class="sep">·</span><span><kbd>←</kbd> <kbd>→</kbd> <kbd>↑</kbd> <kbd>↓</kbd> queue</span><span class="sep">·</span><span><kbd>Z</kbd> half · <kbd>B</kbd> build · <kbd>E</kbd> undo · <kbd>Q</kbd> clear</span></div>
          <div class="action-bar">
            <span data-action-hint>Select a blue tile, then an adjacent destination.</span>
            <div><button type="button" data-half>HALF · Z</button><button type="button" data-build disabled>BUILD · B</button><button type="button" data-policy>POLICY MAP</button></div>
          </div>
          <div class="policy-readout"><span>LAST G08 ACTION</span><b data-policy-action>waiting for model…</b></div>
        </main>
      </div>`;

    this.board = select(root, "[data-board]");
    this.modeButtons = root.querySelectorAll<HTMLButtonElement>("[data-mode]");
    this.pauseButton = select(root, "[data-pause]");
    this.stepButton = select(root, "[data-step]");
    this.status = select(root, "[data-status]");
    this.turn = select(root, "[data-turn]");
    this.names = [select(root, "[data-name-0]"), select(root, "[data-name-1]")];
    this.army = [select(root, "[data-army-0]"), select(root, "[data-army-1]")];
    this.land = [select(root, "[data-land-0]"), select(root, "[data-land-1]")];
    this.buildButton = select(root, "[data-build]");
    this.halfButton = select(root, "[data-half]");
    this.overlayButton = select(root, "[data-policy]");
    this.endScreen = select(root, "[data-end]");
    this.scoreRows = [select(root, "[data-score-row-0]"), select(root, "[data-score-row-1]")];
    this.transportFeedback = select(root, "[data-transport-feedback]");
    this.createBoardTiles();
    if (this.mode === "human") this.opponentSeat = new ChampionSeat(() => this.updateStatus());
    this.bindEvents();
    this.applyMode();
    this.startTimer();
    this.render();
  }

  private bindEvents(): void {
    /** Connect game controls, canvas selection, and keyboard actions. */
    this.modeButtons.forEach((button) => button.addEventListener("click", () => {
      const mode = button.dataset.mode;
      if (mode === "human" || mode === "selfplay") {
        this.mode = mode;
        this.newGame();
      }
    }));
    select<HTMLButtonElement>(this.root, "[data-new]").addEventListener("click", () => this.newGame());
    select<HTMLButtonElement>(this.root, "[data-rematch]").addEventListener("click", () => this.newGame());
    this.pauseButton.addEventListener("click", () => {
      this.running = !this.running;
      this.pauseButton.textContent = this.running ? "❚❚" : "▶";
      this.pauseButton.title = this.running ? "Pause" : "Play";
      this.stepButton.disabled = this.running;
      this.showTransportFeedback(this.running ? "▶" : "❚❚", !this.running);
    });
    this.stepButton.addEventListener("click", () => { void this.tick(); });
    select<HTMLSelectElement>(this.root, "[data-speed]").addEventListener("change", (event) => {
      this.tps = Number((event.target as HTMLSelectElement).value);
      this.startTimer();
    });
    this.board.addEventListener("click", (event) => this.handleBoardClick(event));
    this.halfButton.addEventListener("click", () => this.toggleHalf());
    this.buildButton.addEventListener("click", () => this.queueBuild());
    this.overlayButton.addEventListener("click", () => {
      this.policyOverlay = !this.policyOverlay;
      this.overlayButton.classList.toggle("active", this.policyOverlay);
      this.render();
    });
    addEventListener("keydown", (event: KeyboardEvent) => this.handleKey(event));
  }

  private createBoardTiles(): void {
    /** Rebuild the tile DOM for the current archived map dimensions. */
    this.board.replaceChildren();
    this.tiles.length = 0;
    this.tileNumbers.length = 0;
    this.tileArrows.length = 0;
    this.board.style.setProperty("--cols", String(this.state.cols));
    this.board.style.setProperty("--rows", String(this.state.rows));
    for (let cell = 0; cell < this.state.rows * this.state.cols; cell += 1) {
      const tile = document.createElement("button");
      const icon = document.createElement("span");
      const number = document.createElement("span");
      const arrow = document.createElement("span");
      tile.type = "button";
      tile.className = "tile fog";
      tile.dataset.cell = String(cell);
      tile.setAttribute("aria-label", `cell ${Math.floor(cell / this.state.cols)}, ${cell % this.state.cols}`);
      icon.className = "icon";
      number.className = "num";
      arrow.className = "queue-arrow";
      tile.append(icon, number, arrow);
      this.board.append(tile);
      this.tiles.push(tile);
      this.tileNumbers.push(number);
      this.tileArrows.push(arrow);
    }
  }

  private showTransportFeedback(symbol: "▶" | "❚❚", persist: boolean): void {
    /** Dim the board until resumed; the play feedback then fades away. */
    const glyph = select<HTMLElement>(this.transportFeedback, "span");
    glyph.textContent = symbol;
    this.transportFeedback.getAnimations().forEach((animation: Animation) => animation.cancel());
    if (persist) {
      this.transportFeedback.style.opacity = "1";
      this.transportFeedback.animate(
        [{ opacity: 0 }, { opacity: 1 }],
        { duration: 180, easing: "ease-out" },
      );
      return;
    }
    this.transportFeedback.style.opacity = "0";
    this.transportFeedback.animate(
      [{ opacity: 1 }, { opacity: 1, offset: 0.35 }, { opacity: 0 }],
      { duration: 520, easing: "ease-out" },
    );
  }

  private newGame(): void {
    /** Reset live human play or select another offline self-play trajectory. */
    this.opponentSeat?.dispose();
    this.opponentSeat = null;
    if (this.mode === "selfplay") this.loadSelfplayReplay();
    else {
      this.state = createState(randomCompetitionMap(), false);
      this.opponentSeat = new ChampionSeat(() => this.updateStatus());
    }
    this.createBoardTiles();
    this.humanSeen.fill(0);
    this.humanMountains.fill(0);
    this.humanCastles.fill(0);
    this.selected = this.state.generals[0];
    this.humanQueue.length = 0;
    this.lastDecision = [null, null];
    this.running = true;
    this.pauseButton.textContent = "❚❚";
    this.pauseButton.title = "Pause";
    this.stepButton.disabled = true;
    this.transportFeedback.style.opacity = "0";
    this.endScreen.hidden = true;
    this.applyMode();
    this.render();
  }

  private loadSelfplayReplay(): void {
    /** Select one precomputed g08-vs-g08 trace and restore its initial state. */
    const randomValue = new Uint32Array(1);
    crypto.getRandomValues(randomValue);
    this.selfplayReplay = SELFPLAY_REPLAYS[randomValue[0]! % SELFPLAY_REPLAYS.length]!;
    this.selfplayCursor = 0;
    this.state = createState(competitionMapAt(this.selfplayReplay.mapIndex), false);
  }

  private applyMode(): void {
    /** Update labels and human-only controls for the active matchup. */
    const human = this.mode === "human";
    this.names[0].textContent = human ? "YOU" : "G08 · BLUE";
    this.names[1].textContent = "G08 · RED";
    select<HTMLElement>(this.root, "[data-match-meta]").textContent = human ?
      "YOU vs G08 · live competition sandbox" : "G08 vs G08 · independent fog memories";
    this.root.classList.toggle("human-mode", human);
    this.overlayButton.disabled = !human;
    if (!human) {
      this.policyOverlay = false;
      this.overlayButton.classList.remove("active");
    }
    this.modeButtons.forEach((button) => button.classList.toggle("active", button.dataset.mode === this.mode));
    select<HTMLElement>(this.root, "[data-action-hint]").textContent = human ?
      "Select a blue tile, then an adjacent destination. Arrow keys move · Z sends half · B builds." :
      "Two independent copies of g08 act from separate fog-memory states.";
    this.updateStatus();
  }

  private startTimer(): void {
    /** Restart the bounded async tick timer at the selected playback speed. */
    if (this.timer !== null) clearInterval(this.timer);
    this.timer = window.setInterval(() => {
      if (this.running) void this.tick();
    }, 1000 / this.tps);
  }

  private async tick(): Promise<void> {
    /** Collect simultaneous actions, step once, and redraw without overlap. */
    if (this.busy || this.state.winner !== -1) return;
    if (this.mode === "selfplay") {
      const actions = this.selfplayReplay.actions[this.selfplayCursor];
      if (actions === undefined) throw new Error("Self-play replay ended before the recorded terminal state");
      this.lastDecision = [
        { action: actions[0], actionIndex: 0, logits: EMPTY_LOGITS },
        { action: actions[1], actionIndex: 0, logits: EMPTY_LOGITS },
      ];
      this.selfplayCursor += 1;
      this.state = step(this.state, actions);
      this.render();
      if (this.state.winner !== -1) this.finish();
      return;
    }
    const opponent = this.opponentSeat;
    if (opponent === null || opponent.status !== "ready") return;
    this.busy = true;
    const snapshot = this.state;
    // Dequeue before asynchronous opponent inference. Inputs added while this
    // tick is in flight belong to the next tick and must remain in the queue.
    const queuedHumanAction = this.humanQueue.shift() ?? PASS_ACTION;
    const p0 = Promise.resolve<PolicyDecision>({ action: queuedHumanAction, actionIndex: 0, logits: EMPTY_LOGITS });
    const p1 = opponent.act(snapshot, 1);
    try {
      const decisions = await Promise.all([p0, p1]);
      this.lastDecision = decisions;
      this.state = step(snapshot, [decisions[0].action, decisions[1].action]);
      if (this.mode === "human") {
        const succeeded = this.humanActionSucceeded(decisions[0].action, snapshot);
        if (!succeeded) this.humanQueue.length = 0;
        if (!succeeded || this.humanQueue.length === 0) this.recoverHumanCursor(decisions[0].action);
        this.showCursorHint();
      }
      this.render();
      if (this.state.winner !== -1) this.finish();
    } finally {
      this.busy = false;
    }
  }

  private humanActionSucceeded(action: Action, before: CompetitionState): boolean {
    /** Decide whether a queued action reached a state from which its path may continue. */
    if (action[0] === 1) return true;
    if (action[0] === 2) {
      const cell = action[1] * this.state.cols + action[2];
      return this.state.owners[cell] === 0 && this.state.castles[cell] === 1;
    }
    if (action[1] < 0 || action[1] >= before.rows || action[2] < 0 || action[2] >= before.cols) return false;
    const source = action[1] * before.cols + action[2];
    const [dr, dc] = DIRECTIONS[action[3]]!;
    const row = action[1] + dr;
    const col = action[2] + dc;
    if (row < 0 || row >= before.rows || col < 0 || col >= before.cols) return false;
    const target = row * before.cols + col;
    if (before.owners[source] !== 0 || before.armies[source]! <= 1 || before.mountains[target] === 1) return false;
    return this.state.owners[row * this.state.cols + col] === 0;
  }

  private nearestOwnedCell(origin: number): number | null {
    /** Return the closest surviving human tile with stable row-major tie-breaking. */
    const originRow = Math.floor(origin / this.state.cols);
    const originCol = origin % this.state.cols;
    let closest: number | null = null;
    let closestDistance = Infinity;
    for (let cell = 0; cell < this.state.owners.length; cell += 1) {
      if (this.state.owners[cell] !== 0) continue;
      const row = Math.floor(cell / this.state.cols);
      const col = cell % this.state.cols;
      const distance = Math.abs(row - originRow) + Math.abs(col - originCol);
      if (distance < closestDistance) {
        closest = cell;
        closestDistance = distance;
      }
    }
    return closest;
  }

  private recoverHumanCursor(action: Action): void {
    /** Keep the cursor usable after simultaneous combat changes ownership. */
    if (action[0] === 0) {
      const source = action[1] * this.state.cols + action[2];
      const [dr, dc] = DIRECTIONS[action[3]]!;
      const destination = (action[1] + dr) * this.state.cols + action[2] + dc;
      if (this.state.owners[destination] === 0) {
        this.selected = destination;
        return;
      }
      if (this.state.owners[source] === 0) {
        this.selected = source;
        return;
      }
      this.selected = this.nearestOwnedCell(destination);
      return;
    }
    if (this.selected !== null && this.state.owners[this.selected] === 0) return;
    this.selected = this.nearestOwnedCell(this.selected ?? this.state.generals[0]);
  }

  private showCursorHint(): void {
    /** Describe the persistent cursor after one queued action resolves. */
    const hint = select<HTMLElement>(this.root, "[data-action-hint]");
    if (this.humanQueue.length > 0) {
      hint.textContent = `${this.humanQueue.length} move${this.humanQueue.length === 1 ? "" : "s"} queued · Q clear · E undo`;
      return;
    }
    if (this.selected === null) {
      hint.textContent = "No owned tiles remain.";
      return;
    }
    const row = Math.floor(this.selected / this.state.cols);
    const col = this.selected % this.state.cols;
    hint.textContent = `Cursor: (${row},${col}) · choose an adjacent destination.`;
  }

  private finish(): void {
    /** Show the terminal result without hiding the final board. */
    this.running = false;
    const title = select<HTMLElement>(this.endScreen, "strong");
    const subtitle = select<HTMLElement>(this.endScreen, "span");
    if (this.state.winner === -2) title.textContent = "DRAW";
    else if (this.mode === "human") title.textContent = this.state.winner === 0 ? "YOU WIN" : "G08 WINS";
    else title.textContent = this.state.winner === 0 ? "BLUE G08 WINS" : "RED G08 WINS";
    subtitle.textContent = `turn ${this.state.turn} · new game resets both fog memories`;
    this.endScreen.hidden = false;
  }

  private updateStatus(): void {
    /** Reflect replay mode or the single live opponent worker in the header. */
    if (this.mode === "selfplay") {
      this.status.textContent = "g08 replay ready · zero inference";
      this.status.parentElement?.classList.add("ready");
      return;
    }
    const opponent = this.opponentSeat;
    if (opponent?.status === "error") {
      this.status.textContent = `model error · ${opponent.error}`;
      this.status.parentElement?.classList.add("error");
    } else if (opponent?.status === "ready") {
      this.status.textContent = "g08 ready · exact ONNX export";
      this.status.parentElement?.classList.add("ready");
    } else {
      this.status.textContent = "loading 2.72M parameters…";
    }
  }

  private handleBoardClick(event: MouseEvent): void {
    /** Select an owned source or append one adjacent step to the move queue. */
    if (this.mode !== "human") return;
    const target = event.target as HTMLElement;
    const tile = target.closest<HTMLElement>("[data-cell]");
    if (tile === null) return;
    const cell = Number(tile.dataset.cell);
    const row = Math.floor(cell / this.state.cols);
    const col = cell % this.state.cols;
    if (this.selected === null) {
      if (this.state.owners[cell] === 0) this.selected = cell;
      this.updateActionButtons();
      this.render();
      return;
    }
    const sourceRow = Math.floor(this.selected / this.state.cols);
    const sourceCol = this.selected % this.state.cols;
    const direction = DIRECTIONS.findIndex(([dr, dc]) => sourceRow + dr === row && sourceCol + dc === col);
    if (direction >= 0) {
      this.queueMove(direction);
    } else if (this.state.owners[cell] === 0) {
      this.humanQueue.length = 0;
      this.selected = cell;
    }
    this.updateActionButtons();
    this.render();
  }

  private handleKey(event: KeyboardEvent): void {
    /** Support the original keyboard move workflow and build shortcut. */
    if (this.mode !== "human") return;
    const key = event.key.toLowerCase();
    if (key === "z") {
      this.toggleHalf();
      event.preventDefault();
      return;
    }
    if (key === "b") {
      this.queueBuild();
      event.preventDefault();
      return;
    }
    if (key === "q") {
      this.clearHumanQueue();
      event.preventDefault();
      return;
    }
    if (key === "e" || event.key === "Backspace") {
      this.undoHumanQueue();
      event.preventDefault();
      return;
    }
    const directions: Record<string, number> = { ArrowUp: 0, w: 0, ArrowDown: 1, s: 1, ArrowLeft: 2, a: 2, ArrowRight: 3, d: 3 };
    const direction = directions[event.key];
    if (direction === undefined || this.selected === null) return;
    this.queueMove(direction);
    event.preventDefault();
  }

  private queueMove(direction: number): void {
    /** Append one in-bounds move and advance the planning cursor immediately. */
    if (this.selected === null) return;
    const row = Math.floor(this.selected / this.state.cols);
    const col = this.selected % this.state.cols;
    const [dr, dc] = DIRECTIONS[direction]!;
    const destinationRow = row + dr;
    const destinationCol = col + dc;
    if (destinationRow < 0 || destinationRow >= this.state.rows ||
        destinationCol < 0 || destinationCol >= this.state.cols) {
      select<HTMLElement>(this.root, "[data-action-hint]").textContent = "Illegal edge move · cursor unchanged.";
      return;
    }
    if (this.humanQueue.length === 0 &&
        (this.state.owners[this.selected] !== 0 || this.state.armies[this.selected]! <= 1)) {
      select<HTMLElement>(this.root, "[data-action-hint]").textContent = "Source cannot move yet · cursor unchanged.";
      return;
    }
    const paddedTarget = destinationRow * PAD_TO + destinationCol;
    if (this.humanMountains[paddedTarget] === 1) {
      select<HTMLElement>(this.root, "[data-action-hint]").textContent = "Mountain blocks the move · cursor unchanged.";
      return;
    }
    this.humanQueue.push([0, row, col, direction, this.half ? 1 : 0]);
    this.selected = destinationRow * this.state.cols + destinationCol;
    select<HTMLElement>(this.root, "[data-action-hint]").textContent =
      `${this.humanQueue.length} move${this.humanQueue.length === 1 ? "" : "s"} queued · Q clear · E undo`;
    this.render();
  }

  private clearHumanQueue(): void {
    /** Clear every pending move and restore the cursor to the first source. */
    const first = this.humanQueue[0];
    this.humanQueue.length = 0;
    if (first !== undefined) {
      const source = first[1] * this.state.cols + first[2];
      this.selected = this.state.owners[source] === 0 ? source : this.nearestOwnedCell(source);
    }
    this.showCursorHint();
    this.render();
  }

  private undoHumanQueue(): void {
    /** Remove the newest pending action and return the cursor to its source. */
    const action = this.humanQueue.pop();
    if (action !== undefined) this.selected = action[1] * this.state.cols + action[2];
    this.showCursorHint();
    this.render();
  }

  private toggleHalf(): void {
    /** Toggle whether the next human move sends half or all but one army. */
    this.half = !this.half;
    this.halfButton.classList.toggle("active", this.half);
    this.halfButton.textContent = this.half ? "HALF ON · Z" : "HALF · Z";
  }

  private queueBuild(): void {
    /** Queue a legal human castle action from the selected source. */
    if (this.selected === null || this.buildButton.disabled || this.humanQueue.length > 0) return;
    const row = Math.floor(this.selected / this.state.cols);
    const col = this.selected % this.state.cols;
    this.humanQueue.push([2, row, col, 0, 0]);
    select<HTMLElement>(this.root, "[data-action-hint]").textContent = `Queued castle at (${row},${col})`;
    this.render();
  }

  private updateActionButtons(): void {
    /** Expose the current build price only when a human source is selected. */
    if (this.selected === null) {
      this.buildButton.disabled = true;
      this.buildButton.textContent = "BUILD · B";
      return;
    }
    const row = Math.floor(this.selected / this.state.cols);
    const col = this.selected % this.state.cols;
    const source = this.selected;
    const price = buildCost(this.state, 0, row, col);
    const isGeneral = this.state.generals[0] === source || this.state.generals[1] === source;
    this.buildButton.disabled = this.state.owners[source] !== 0 || this.state.castles[source] === 1 || isGeneral || this.state.armies[source]! < price;
    this.buildButton.textContent = `BUILD ${price} · B`;
  }

  private render(): void {
    /** Draw the official-style board, scores, selection, and optional policy saliency. */
    const viewPlayer: Player | null = this.mode === "human" ? 0 : null;
    const visible = viewPlayer === null ? new Uint8Array(this.state.owners.length).fill(1) : visibility(this.state, viewPlayer);
    for (let row = 0; row < this.state.rows; row += 1) {
      for (let col = 0; col < this.state.cols; col += 1) {
        const source = row * this.state.cols + col;
        const padded = row * PAD_TO + col;
        if (visible[source] === 1) {
          this.humanSeen[padded] = 1;
          if (this.state.mountains[source] === 1) this.humanMountains[padded] = 1;
          if (this.state.castles[source] === 1) this.humanCastles[padded] = 1;
        }
      }
    }

    const policy = this.lastDecision[1]?.logits ?? null;
    let policyMin = Infinity;
    let policyMax = -Infinity;
    if (this.policyOverlay && policy !== null) {
      for (let position = 0; position < PAD_TO * PAD_TO; position += 1) {
        let best = -Infinity;
        for (let kind = 0; kind < 10; kind += 1) best = Math.max(best, policy[kind * PAD_TO * PAD_TO + position]!);
        if (best > -1e8) {
          policyMin = Math.min(policyMin, best);
          policyMax = Math.max(policyMax, best);
        }
      }
    }
    for (const arrow of this.tileArrows) arrow.textContent = "";
    for (const action of this.humanQueue) {
      if (action[0] !== 0) continue;
      const source = action[1] * this.state.cols + action[2];
      this.tileArrows[source]!.textContent = ["↑", "↓", "←", "→"][action[3]]!;
    }

    for (let row = 0; row < this.state.rows; row += 1) {
      for (let col = 0; col < this.state.cols; col += 1) {
        const source = row * this.state.cols + col;
        const padded = row * PAD_TO + col;
        const tile = this.tiles[source]!;
        const number = this.tileNumbers[source]!;
        const seenNow = visible[source] === 1;
        const mountain = seenNow ? this.state.mountains[source] === 1 : this.humanMountains[padded] === 1;
        const castle = seenNow ? this.state.castles[source] === 1 : this.humanCastles[padded] === 1;
        const isGeneral = this.state.generals[0] === source || this.state.generals[1] === source;
        const showGeneral = isGeneral && (seenNow || this.state.owners[source] === viewPlayer || this.state.winner !== -1);
        const classes = ["tile"];
        if (!seenNow) classes.push("fog");
        else if (this.state.owners[source] === 0) classes.push("blue");
        else if (this.state.owners[source] === 1) classes.push("red");
        else if (mountain) classes.push("mountain");
        else if (castle) classes.push("neutral-castle");
        else classes.push("neutral");
        if (mountain) classes.push(seenNow ? "has-mountain" : "fog-mountain", "has-mountain");
        else if (showGeneral) classes.push("has-general");
        else if (castle) classes.push("has-castle");
        if (this.selected === source) classes.push("selected");
        tile.className = classes.join(" ");
        number.textContent = seenNow && !mountain &&
          (this.state.owners[source]! >= 0 || this.state.armies[source]! > 0) ? String(this.state.armies[source]!) : "";
        tile.style.removeProperty("--policy-alpha");

        if (this.policyOverlay && policy !== null && seenNow) {
          let best = -Infinity;
          for (let kind = 0; kind < 10; kind += 1) best = Math.max(best, policy[kind * PAD_TO * PAD_TO + padded]!);
          if (best > -1e8 && policyMax > policyMin) {
            const strength = (best - policyMin) / (policyMax - policyMin);
            tile.style.setProperty("--policy-alpha", String(0.08 + strength * 0.58));
          }
        }
      }
    }

    this.turn.textContent = String(this.state.turn);
    select<HTMLInputElement>(this.root, "[data-progress]").value = String(this.state.turn);
    const scores = [totals(this.state, 0), totals(this.state, 1)] as const;
    for (const player of [0, 1] as const) {
      this.land[player].textContent = String(scores[player][0]);
      this.army[player].textContent = String(scores[player][1]);
    }
    const leader = scores[0][0] === scores[1][0] ? (scores[0][1] >= scores[1][1] ? 0 : 1) : (scores[0][0] > scores[1][0] ? 0 : 1);
    this.scoreRows[0].classList.toggle("is-leader", leader === 0);
    this.scoreRows[1].classList.toggle("is-leader", leader === 1);
    const last = this.lastDecision[1];
    select<HTMLElement>(this.root, "[data-policy-action]").textContent = last === null ? "waiting for model…" : this.describeAction(last.action);
    this.updateActionButtons();
  }

  private describeAction(action: Action): string {
    /** Format one protocol action for the live policy readout. */
    if (action[0] === 1) return "pass";
    if (action[0] === 2) return `build at (${action[1]}, ${action[2]})`;
    return `${action[4] === 1 ? "half" : "full"} from (${action[1]}, ${action[2]}) ${["up", "down", "left", "right"][action[3]]}`;
  }
}
