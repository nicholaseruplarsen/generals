(() => {
  const LAYERS = ["INPUT", "STEM", "BLOCK 2", "GLOBAL 1", "BLOCK 5", "GLOBAL 2", "POLICY"];
  const CONCEPTS = ["FRONTIER", "MOTION", "THREAT", "ECONOMY", "FOG AGE", "GENERAL", "BUILD", "DEADLINE"];
  const CONCEPT_COPY = {
    FRONTIER: "Local ownership boundaries are available immediately from the observation planes.",
    MOTION: "Seven delta frames make army flow explicit before the first learned layer.",
    THREAT: "A large remote stack becomes globally available only after a pooled broadcast.",
    ECONOMY: "Broadcast score and pooled occupancy support a whole-board resource comparison.",
    "FOG AGE": "Time-since-seen is supplied directly; later layers can combine staleness with threat.",
    GENERAL: "Candidate locations narrow through the belief channels and become action-relevant late.",
    BUILD: "Build preference requires local affordability and global strategic context.",
    DEADLINE: "Turn phase is explicit, but its effect should sharpen near deathtouch.",
  };
  const PHASE_WEIGHTS = {
    opening: [0.18, 0.22, 0.08, 0.36, 0.25, 0.12, 0.30, 0.04],
    contact: [0.58, 0.63, 0.78, 0.51, 0.66, 0.54, 0.42, 0.18],
    endgame: [0.71, 0.57, 0.92, 0.48, 0.59, 0.86, 0.24, 0.95],
  };

  /** Return a bounded schematic loading for one layer, concept, and game phase. */
  function lensValue(phase, layerIndex, conceptIndex) {
    const base = PHASE_WEIGHTS[phase][conceptIndex];
    const depthGain = layerIndex * (conceptIndex === 0 || conceptIndex === 1 ? 0.035 : 0.07);
    const broadcastGain = (layerIndex >= 3 ? 0.16 : 0) + (layerIndex >= 5 ? 0.10 : 0);
    const globalConcept = conceptIndex === 2 || conceptIndex === 3 || conceptIndex === 5 || conceptIndex === 7;
    const policyCompression = layerIndex === 6 && conceptIndex < 2 ? -0.12 : 0;
    return Math.max(0.03, Math.min(0.98, base + depthGain + (globalConcept ? broadcastGain : 0) + policyCompression));
  }

  /** Mount the layer-by-concept matrix and phase controls. */
  function initWorkspaceLens(root) {
    const matrix = root.querySelector("[data-lens-matrix]");
    const coordinate = root.querySelector("[data-lens-coordinate]");
    const value = root.querySelector("[data-lens-value]");
    const copy = root.querySelector("[data-lens-copy]");
    let phase = "opening";
    let selectedLayer = 1;
    let selectedConcept = 0;

    const render = () => {
      matrix.replaceChildren();
      matrix.style.setProperty("--lens-columns", String(CONCEPTS.length + 1));
      const corner = document.createElement("span");
      corner.className = "lens-axis lens-corner";
      corner.textContent = "LAYER / SIGNAL";
      matrix.append(corner);
      CONCEPTS.forEach((concept) => {
        const label = document.createElement("span");
        label.className = "lens-axis lens-column";
        label.textContent = concept;
        matrix.append(label);
      });
      LAYERS.forEach((layer, layerIndex) => {
        const label = document.createElement("span");
        label.className = "lens-axis lens-row";
        label.textContent = layer;
        matrix.append(label);
        CONCEPTS.forEach((concept, conceptIndex) => {
          const loading = lensValue(phase, layerIndex, conceptIndex);
          const cell = document.createElement("button");
          cell.type = "button";
          cell.className = "lens-cell";
          cell.classList.toggle("selected", layerIndex === selectedLayer && conceptIndex === selectedConcept);
          cell.style.setProperty("--loading", loading.toFixed(3));
          cell.setAttribute("aria-label", `${layer}, ${concept}, schematic loading ${loading.toFixed(2)}`);
          cell.addEventListener("click", () => {
            selectedLayer = layerIndex;
            selectedConcept = conceptIndex;
            render();
          });
          matrix.append(cell);
        });
      });
      const selectedValue = lensValue(phase, selectedLayer, selectedConcept);
      coordinate.textContent = `${LAYERS[selectedLayer]} × ${CONCEPTS[selectedConcept]}`;
      value.textContent = selectedValue.toFixed(2);
      copy.textContent = CONCEPT_COPY[CONCEPTS[selectedConcept]];
    };

    root.querySelectorAll("[data-lens-phase]").forEach((button) => {
      button.addEventListener("click", () => {
        phase = button.dataset.lensPhase;
        root.querySelectorAll("[data-lens-phase]").forEach((item) => item.classList.toggle("active", item === button));
        render();
      });
    });
    render();
  }

  /** Mount the spatial receptive-field and broadcast topology explainer. */
  function initBroadcastLab(root) {
    const board = root.querySelector("[data-broadcast-board]");
    const depth = root.querySelector("[data-broadcast-depth]");
    const depthValue = root.querySelector("[data-broadcast-depth-value]");
    const toggle = root.querySelector("[data-broadcast-toggle]");
    const reach = root.querySelector("[data-broadcast-reach]");
    const copy = root.querySelector("[data-broadcast-copy]");
    const meter = root.querySelector("[data-broadcast-meter]");
    const size = 15;
    const alarm = { row: 2, col: 12 };
    let target = { row: 12, col: 2 };
    let broadcast = false;

    for (let row = 0; row < size; row += 1) {
      for (let col = 0; col < size; col += 1) {
        const cell = document.createElement("button");
        cell.type = "button";
        cell.dataset.row = String(row);
        cell.dataset.col = String(col);
        cell.setAttribute("aria-label", `Select decision tile ${row}, ${col}`);
        cell.addEventListener("click", () => {
          target = { row, col };
          render();
        });
        board.append(cell);
      }
    }

    const render = () => {
      const blocks = Number(depth.value);
      const radius = blocks;
      const alarmVisible = Math.abs(alarm.row - target.row) <= radius && Math.abs(alarm.col - target.col) <= radius;
      board.querySelectorAll("button").forEach((cell) => {
        const row = Number(cell.dataset.row);
        const col = Number(cell.dataset.col);
        const local = Math.abs(row - target.row) <= radius && Math.abs(col - target.col) <= radius;
        cell.className = "";
        cell.classList.toggle("local", local);
        cell.classList.toggle("broadcast", broadcast && !local);
        cell.classList.toggle("target", row === target.row && col === target.col);
        cell.classList.toggle("alarm", row === alarm.row && col === alarm.col);
      });
      depthValue.textContent = String(blocks);
      toggle.textContent = broadcast ? "Global broadcast on" : "Global broadcast off";
      toggle.setAttribute("aria-pressed", String(broadcast));
      reach.textContent = broadcast ? "Whole-board summary" : `${blocks * 2 + 1}×${blocks * 2 + 1} local view`;
      if (broadcast) {
        copy.textContent = "The distant alarm contributes through max pooling, even though its location is compressed away.";
        meter.style.width = "100%";
      } else if (alarmVisible) {
        copy.textContent = "The alarm lies inside the local receptive field and can reach this decision directly.";
        meter.style.width = "76%";
      } else {
        copy.textContent = "The far threat cannot affect this tile through the residual trunk yet.";
        meter.style.width = `${12 + blocks * 7}%`;
      }
    };

    depth.addEventListener("input", render);
    toggle.addEventListener("click", () => {
      broadcast = !broadcast;
      render();
    });
    render();
  }

  /** Convert unnormalized action scores to percentages. */
  function softmax(scores) {
    const maximum = Math.max(...scores);
    const exponentials = scores.map((score) => Math.exp(score - maximum));
    const total = exponentials.reduce((sum, score) => sum + score, 0);
    return exponentials.map((score) => (score / total) * 100);
  }

  /** Mount the feature-ablation workbench and action response bars. */
  function initInterventionLab(root) {
    const features = { threat: true, economy: true, certainty: true };
    const actions = ["attack", "defend", "expand", "build", "pass"];
    const base = [0.35, 0.25, 0.85, -0.10, -0.55];
    const effects = {
      threat: [0.25, 1.25, -0.75, -0.45, 0.20],
      economy: [0.45, -0.20, 0.65, 0.90, -0.40],
      certainty: [1.10, -0.35, -0.20, -0.15, -0.45],
    };
    const label = root.querySelector("[data-intervention-label]");
    const copy = root.querySelector("[data-intervention-copy]");

    const render = () => {
      const enabled = Object.keys(features).filter((feature) => features[feature]);
      const scores = base.map((score, actionIndex) => enabled.reduce(
        (total, feature) => total + effects[feature][actionIndex],
        score,
      ));
      const probabilities = softmax(scores);
      actions.forEach((action, index) => {
        const row = root.querySelector(`[data-action="${action}"]`);
        row.querySelector("b").textContent = `${probabilities[index].toFixed(1)}%`;
        row.querySelector("em").style.width = `${probabilities[index]}%`;
      });
      if (enabled.length === 3) {
        label.textContent = "NATURAL FORWARD PASS";
        copy.textContent = "All three candidate signals are present. Remove one and inspect which actions actually move.";
      } else if (enabled.length === 0) {
        label.textContent = "ALL CANDIDATES ABLATED";
        copy.textContent = "The remaining distribution reflects the local baseline; this is the intervention control.";
      } else {
        const removed = Object.keys(features).filter((feature) => !features[feature]).join(" + ");
        label.textContent = `ABLATION · ${removed.toUpperCase()}`;
        copy.textContent = "A causal claim would compare this shift across matched real states and report uncertainty.";
      }
    };

    root.querySelectorAll("[data-feature]").forEach((button) => {
      button.addEventListener("click", () => {
        const feature = button.dataset.feature;
        features[feature] = !features[feature];
        button.classList.toggle("active", features[feature]);
        button.setAttribute("aria-pressed", String(features[feature]));
        render();
      });
    });
    render();
  }

  document.querySelectorAll("[data-workspace-lens]").forEach(initWorkspaceLens);
  document.querySelectorAll("[data-broadcast-lab]").forEach(initBroadcastLab);
  document.querySelectorAll("[data-intervention-lab]").forEach(initInterventionLab);
})();
