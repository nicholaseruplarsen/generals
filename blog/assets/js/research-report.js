(() => {
  const ARCHITECTURE = {
    input: ["45 × 21 × 21", "Present state, seven-frame motion history, fog memory, inferred general candidates, and phase clocks."],
    stem: ["192 × 21 × 21", "A 1×1 stem lifts each tile; a 512-turn opponent army/land tape is broadcast across the board."],
    local: ["6 residual blocks", "GroupNorm → SiLU → 3×3 convolution. Identity skips preserve gradient flow while tactics propagate locally."],
    global: ["2 global injections", "Mean and max pool all 441 cells, pass through an MLP, then broadcast-add in O(board area)."],
    policy: ["4,410 logits", "Ten action planes per tile. A rules-derived mask removes impossible moves before sampling."],
    value: ["128 value bins", "Globally pooled features predict an HL-Gauss return distribution over [−1.6, 1.6]."],
  };

  const RULES = {
    fog: ["Observation is history-dependent", "Enemy stacks vanish outside a one-cell visibility halo. Two identical visible frames can require different actions because their hidden histories differ."],
    collision: ["Actions resolve simultaneously", "Chasing, reinforcing, then smaller source army decide move order. The opponent reacts to the same prior state, not to your resolved move."],
    economy: ["Building is delayed commitment", "A castle costs 35 plus a crowding surcharge. It weakens the current stack but produces army every second turn thereafter."],
    deadline: ["The objective changes at turn 800", "Deathtouch makes contact with the enemy general lethal regardless of defense. Turn 1200 is a hard draw."],
  };

  const CHANNELS = {
    present: ["21 planes", "The current legal observation: normalized armies, ownership masks, structures, visibility, scores, board shape, and phase clocks."],
    memory: ["10 planes", "Last-seen enemy strength, logarithmic time-since-seen, candidate general locations, remembered terrain, and deadline features."],
    motion: ["14 planes", "Seven own-army deltas and seven enemy-army deltas expose local velocity without recurrent backpropagation through time."],
  };

  const EXPERIMENTS = {
    d4: {
      status: "WORKED · G03",
      title: "Put symmetry in the behavior policy",
      body: "The first version rotated samples only inside PPO updates. Because the CNN is not equivariant, the stored action probability no longer described the policy that generated the trajectory. The correction sampled one D4 frame per episode and acted inside it.",
      inference: "Inference: symmetry helped once PPO's on-policy likelihood ratio was valid.",
      label: "POLICY RATIO",
      bars: [["update-only", 23, "bad"], ["rollout-D4", 78, "good"]],
    },
    reuse: {
      status: "WORKED · G04–G08",
      title: "Spend each trajectory more carefully",
      body: "1,024-sample minibatches beat 2,048; two PPO passes beat one; λ=0.9 beat 0.7; an entropy floor of .003 beat .001; top-50% |advantage| selection gained more parent margin than top-25%.",
      inference: "Inference: in this lineage, update quality improved strength per sampled step.",
      label: "PARENT MARGIN",
      bars: [["baseline", 35, "neutral"], ["2 epochs", 61, "good"], ["top-50", 79, "good"]],
    },
    expert: {
      status: "FAILED · 400M+ STEPS",
      title: "A stronger opponent did not create a stronger learner",
      body: "Mixing self-play with a frozen transformer champion produced almost no wins against that anchor. Sparse terminal PPO supplied little information about which of thousands of earlier decisions differed.",
      inference: "Inference: opponent strength cannot substitute for a learnable credit signal.",
      label: "WIN RATE VS TRANSFORMER",
      bars: [["start", 2, "bad"], ["200M", 4, "bad"], ["400M", 3, "bad"]],
    },
    unet: {
      status: "FAILED · DEPLOYMENT",
      title: "The larger U-Net learned, then missed the move budget",
      body: "A multi-scale large-kernel U-Net was behavior-cloned from top-player traces and entered PPO, but repeatedly exceeded the competition's 150 ms deadline.",
      inference: "Inference: deployment latency belongs inside the architecture hypothesis.",
      label: "INFERENCE LATENCY",
      bars: [["g08 CNN", 14, "good"], ["limit", 45, "neutral"], ["U-Net", 92, "bad"]],
    },
    drift: {
      status: "MIXED · SELF-PLAY",
      title: "More training sometimes made the policy worse",
      body: "A 104.9M-step continuation lost to the same recipe at 45.9M steps, 230–268–11. Symmetric play had rewarded a slow, castle-heavy convention that did not improve external strength.",
      inference: "Inference: equilibrium compatibility and fixed-anchor strength can move apart.",
      label: "FIXED-ANCHOR STRENGTH",
      bars: [["45.9M", 67, "good"], ["104.9M", 48, "bad"]],
    },
    build: {
      status: "FAILED · CAUSAL AUXILIARY",
      title: "The build classifier learned; the policy built less",
      body: "Paired force-build/control rollouts raised preference accuracy from ~.33 to ~.82, while natural build rate fell from .324% to .264%. Most interventions shared the same terminal result.",
      inference: "Inference: predictable supervision is not necessarily decision-relevant supervision.",
      label: "NORMALIZED CHANGE",
      bars: [["aux accuracy", 82, "good"], ["build rate", 26, "bad"], ["causal signal", 17, "bad"]],
    },
  };

  function initArchitecture(root) {
    const copy = root.querySelector("[data-arch-copy]");
    root.querySelectorAll("[data-arch]").forEach((button) => {
      button.addEventListener("click", () => {
        root.querySelectorAll("[data-arch]").forEach((item) => item.classList.toggle("active", item === button));
        const [title, body] = ARCHITECTURE[button.dataset.arch];
        copy.innerHTML = `<strong>${title}</strong><span>${body}</span>`;
      });
    });
    const board = root.querySelector(".rf-board");
    const range = root.querySelector("[data-rf-range]");
    const label = root.querySelector("[data-rf-label]");
    for (let index = 0; index < 441; index += 1) board.append(document.createElement("i"));
    const render = () => {
      const blocks = Number(range.value);
      const radius = blocks;
      board.querySelectorAll("i").forEach((cell, index) => {
        const row = Math.floor(index / 21);
        const col = index % 21;
        cell.classList.toggle("seen", Math.abs(row - 10) <= radius && Math.abs(col - 10) <= radius);
        cell.classList.toggle("center", row === 10 && col === 10);
      });
      label.textContent = `${blocks * 2 + 1}×${blocks * 2 + 1}`;
    };
    range.addEventListener("input", render);
    render();
  }

  function initRules(root) {
    const board = root.querySelector("[data-rule-board]");
    const title = root.querySelector("[data-rule-title]");
    const copy = root.querySelector("[data-rule-copy]");
    for (let index = 0; index < 49; index += 1) board.append(document.createElement("i"));
    root.querySelectorAll("[data-rule]").forEach((button) => {
      button.addEventListener("click", () => {
        const rule = button.dataset.rule;
        root.querySelectorAll("[data-rule]").forEach((item) => item.classList.toggle("active", item === button));
        root.dataset.activeRule = rule;
        [title.textContent, copy.textContent] = RULES[rule];
      });
    });
    root.dataset.activeRule = "fog";
  }

  function initChannels(root) {
    root.querySelectorAll("[data-channel]").forEach((button) => {
      button.addEventListener("click", () => {
        root.querySelectorAll("[data-channel]").forEach((item) => item.classList.toggle("active", item === button));
        const [count, copy] = CHANNELS[button.dataset.channel];
        root.querySelector("[data-channel-count]").textContent = count;
        root.querySelector("[data-channel-copy]").textContent = copy;
      });
    });
  }

  function initCredit(root) {
    const delay = root.querySelector("[data-delay]");
    const lambda = root.querySelector("[data-lambda]");
    const bars = root.querySelector("[data-credit-bars]");
    for (let index = 0; index < 24; index += 1) bars.append(document.createElement("i"));
    const render = () => {
      const horizon = Number(delay.value);
      const decay = Number(lambda.value) / 100;
      root.querySelector("[data-delay-value]").textContent = `${horizon} turns`;
      root.querySelector("[data-lambda-value]").textContent = decay.toFixed(2);
      bars.querySelectorAll("i").forEach((bar, index) => {
        const distance = index + 1;
        bar.style.height = `${Math.max(3, Math.pow(decay, distance) * 100)}%`;
        bar.classList.toggle("beyond", distance > horizon);
      });
      const retained = Math.pow(decay, horizon) * 100;
      root.querySelector("[data-credit-readout]").textContent = `${horizon}-turn signal retains ${retained.toFixed(1)}% weight`;
    };
    delay.addEventListener("input", render);
    lambda.addEventListener("input", render);
    render();
  }

  function initExperiments(root) {
    const visual = root.querySelector("[data-experiment-visual]");
    const render = (key) => {
      const experiment = EXPERIMENTS[key];
      root.querySelector("[data-experiment-status]").textContent = experiment.status;
      root.querySelector("[data-experiment-title]").textContent = experiment.title;
      root.querySelector("[data-experiment-body]").textContent = experiment.body;
      root.querySelector("[data-experiment-inference]").textContent = experiment.inference;
      visual.innerHTML = `<span>${experiment.label}</span><div>${experiment.bars.map(([name, value, kind]) => `<figure><i class="${kind}" style="height:${Math.max(3, value)}%"></i><b>${name}</b><em>${value}</em></figure>`).join("")}</div>`;
    };
    root.querySelectorAll("[data-experiment]").forEach((button) => {
      button.addEventListener("click", () => {
        root.querySelectorAll("[data-experiment]").forEach((item) => item.classList.toggle("active", item === button));
        render(button.dataset.experiment);
      });
    });
    render("d4");
  }

  document.querySelectorAll(".architecture-lab").forEach(initArchitecture);
  document.querySelectorAll("[data-rule-lab]").forEach(initRules);
  document.querySelectorAll("[data-channel-atlas]").forEach(initChannels);
  document.querySelectorAll("[data-credit-lab]").forEach(initCredit);
  document.querySelectorAll("[data-experiment-console]").forEach(initExperiments);
})();
