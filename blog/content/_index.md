---
title: "The Denominator Wins"
subtitle: "Thirteen days, 1.76 billion self-play steps, sixty-odd dollars of rented GPUs, and 9th of 114. What I tried, what worked, and which of it I actually believe."
description: "A retrospective on the 2026 Generals bot competition: the sample-efficiency versus throughput tradeoff, two bugs worth more than any architecture change, and why most of my ablations do not survive scrutiny."
date: 2026-08-11
draft: false
author: "Nicholas Erup Larsen"
kicker: "GENERALS · COMPETITION RETROSPECTIVE"
---

In late July I entered a one-week bot competition for **Generals**, an imperfect-information real-time strategy game, run with a $3,000 first prize. I wrote my own self-play training stack in JAX on top of a C/CUDA engine, rented GPUs by the hour on a student's budget, and finished 9th of 114. This is what I tried, in the order I tried it, with the numbers attached — including the experiments that were wrong, the two bugs that mattered more than any architectural idea I had, and the reason I now distrust most of my own ablation results.

<div class="result-strip">
  <div><span>FINAL RANK</span><strong>9<small>/ 114</small></strong></div>
  <div><span>TRAINING</span><strong>1.764B<small> steps</small></strong></div>
  <div><span>POLICY</span><strong>2.72M<small> params</small></strong></div>
  <div><span>INFERENCE</span><strong>20.7<small> ms / 150</small></strong></div>
</div>

The through-line arrived about four days in, and it is arithmetic rather than insight. On a fixed compute budget you are not optimising learning. You are optimising learning *per second*, and almost every intervention that buys you one costs you the other.

Before any of that, the thing itself. This is the exact submitted policy, exported from Equinox to ONNX and running in your browser.

{{< generals-game mode="human" >}}

## The brief is four numbers

Generals is played on a small rectangular grid under fog of war. You own a general that spawns army every other turn; you move stacks of army into adjacent tiles to claim territory, and you win by walking a bigger stack onto the enemy general. You can only see tiles adjacent to something you own, so most of the board is a memory problem rather than a perception problem.

The 2026 competition ruleset differs from public Generals in ways that change training rather than flavour. Maps are rectangular with each axis drawn independently in 18–21, so the network cannot assume a shape. There are no neutral castles on the map: **castles are built, not captured**, at a cost of 35 army plus a crowding surcharge of `max(0, 14 − 2d)` against your nearest existing castle, and building consumes your turn. Generals spawn at least 17 BFS steps apart. From turn 800 mutual contact is *deathtouch* — both generals die, scored as a draw — and the game truncates to a draw at turn 1200.

Those rules are the interesting part of the problem. The four numbers that actually constrained every decision are on the other side of the submission boundary:

<figure class="rep-fig">
  <p class="rep-fig-title">Figure 2 &nbsp;·&nbsp; The submission budget, and how much of it I used</p>
  <div class="rep-scroll">
  <svg viewBox="0 0 1080 300" role="img" aria-label="The competition allows 150 milliseconds per move, one CPU core and 2 gigabytes of memory with no GPU. The final policy used 20.7 milliseconds and about 10 megabytes, leaving 86 percent of the latency budget and 99.5 percent of the memory unused.">
    <defs>
      <pattern id="hatch" width="6" height="6" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
        <line x1="0" y1="0" x2="0" y2="6" stroke="var(--line)" stroke-width="1.6"/>
      </pattern>
      <marker id="ar1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M0 0 L10 5 L0 10 z" fill="var(--faint)"/>
      </marker>
    </defs>
    <!-- ---------- latency ---------- -->
    <text class="ttl" x="0" y="14">TIME PER MOVE</text>
    <rect x="0" y="26" width="860" height="34" fill="url(#hatch)" stroke="var(--line)"/>
    <rect x="0" y="26" width="118.7" height="34" fill="var(--blue)"/>
    <text x="128" y="48" fill="var(--blue)" style="font-size:12.5px">20.7 ms used</text>
    <text x="852" y="48" text-anchor="end" class="mut" style="font-size:11px">150 ms limit</text>
    <line x1="860" y1="20" x2="860" y2="66" stroke="var(--muted)" stroke-width="1.5"/>
    <line x1="123" y1="72" x2="856" y2="72" stroke="var(--faint)" stroke-width="1"
          marker-start="url(#ar1)" marker-end="url(#ar1)"/>
    <text x="490" y="88" text-anchor="middle" class="mut">86% of the latency budget unused</text>
    <!-- ---------- memory ---------- -->
    <text class="ttl" x="0" y="130">MEMORY</text>
    <rect x="0" y="142" width="860" height="34" fill="url(#hatch)" stroke="var(--line)"/>
    <rect x="0" y="142" width="4.4" height="34" fill="var(--blue)"/>
    <text x="14" y="164" fill="var(--blue)" style="font-size:12.5px">10.5 MB of weights</text>
    <text x="852" y="164" text-anchor="end" class="mut" style="font-size:11px">2048 MB cap</text>
    <line x1="860" y1="136" x2="860" y2="182" stroke="var(--muted)" stroke-width="1.5"/>
    <!-- ---------- hardware ---------- -->
    <text class="ttl" x="0" y="228">HARDWARE AT MATCH TIME</text>
    <g transform="translate(0,240)">
      <rect x="0" y="0" width="168" height="34" fill="var(--paper-2)" stroke="var(--line)"/>
      <text x="84" y="22" text-anchor="middle" fill="var(--ink)">1 CPU core</text>
      <rect x="180" y="0" width="168" height="34" fill="var(--paper-2)" stroke="var(--line)"/>
      <text x="264" y="22" text-anchor="middle" fill="var(--ink)">no GPU</text>
      <rect x="360" y="0" width="168" height="34" fill="var(--paper-2)" stroke="var(--line)"/>
      <text x="444" y="22" text-anchor="middle" fill="var(--ink)">no network</text>
      <rect x="540" y="0" width="320" height="34" fill="var(--paper-2)" stroke="var(--line)"/>
      <text x="700" y="22" text-anchor="middle" fill="var(--ink)">50 faults forfeits the game</text>
    </g>
    <!-- right column: the consequence -->
    <line x1="906" y1="0" x2="906" y2="278" stroke="var(--line)"/>
    <text class="ttl" x="926" y="14">CONSEQUENCE</text>
    <text x="926" y="40" style="font-size:11px">Whatever you train</text>
    <text x="926" y="58" style="font-size:11px">must run as a numpy</text>
    <text x="926" y="76" style="font-size:11px">program, single-core,</text>
    <text x="926" y="94" style="font-size:11px">in under 150 ms.</text>
    <text x="926" y="132" style="font-size:11px" fill="var(--red)">The 2.72 M CNN was</text>
    <text x="926" y="150" style="font-size:11px" fill="var(--red)">sized for that limit</text>
    <text x="926" y="168" style="font-size:11px" fill="var(--red)">and undershot it</text>
    <text x="926" y="186" style="font-size:11px" fill="var(--red)">by 7×.</text>
  </svg>
  </div>
  <figcaption><span class="fig-number">Figure 2</span>Match-time limits (hatched) against what the final
    submission actually consumed (solid). Latency was measured on the CPU numpy bot over 663 turns
    with zero action mismatches against the JAX reference and a maximum logit difference of
    8.2&nbsp;×&nbsp;10<sup>−5</sup>. The unused 86% is the clearest thing I got wrong: I sized the
    network against a limit I never approached.</figcaption>
</figure>

I sized the policy at 2.7 million parameters early, on the assumption that inference would be tight, and never revisited it. It was not tight. The final bot spent 20.7 ms of a 150 ms allowance and 10 MB of a 2 GB allowance. There was room for a network several times larger, or for a shallow search on top of the policy, and I spent the week improving the training of a model that was smaller than the rules required. That is a planning error, not a research finding, and it is the first thing I would change.

## One equation

Fix a strength target — say, beating a frozen reference checkpoint by a statistically significant margin. Let \\(N\\) be the number of agent-steps needed to get there and \\(R\\) the agent-steps per second the stack sustains. The thing you actually spend is

<div class="formula">\(T = N / R\)<small>time to a fixed strength target = steps needed ÷ steps per second</small></div>

which is trivial, and which I nonetheless kept reasoning about only one term at a time. Nearly every change I made moved both terms, in opposite directions. A richer opponent mix makes each sample more informative and makes every rollout slower. Reusing more of each rollout in the PPO update raises the work per sample and lowers the samples per second. Reward shaping, auxiliary losses, larger minibatches, augmentation, a bigger network: all of them trade \\(N\\) against \\(R\\).

So the acceptance test for an intervention is not "does it learn in fewer samples". Writing \\(a = N'/N\\) for the fraction of samples the new recipe needs and \\(b = R'/R\\) for the fraction of throughput it retains,

<div class="formula">\(\dfrac{N'}{R'} < \dfrac{N}{R} \quad\Longleftrightarrow\quad \underbrace{1-a}_{\text{sample-efficiency gain}} > \underbrace{1-b}_{\text{throughput cost}}\)<small>the fractional reduction in samples has to exceed the fractional loss in throughput</small></div>

Everything above the diagonal in Figure 3 is worth doing; everything below it is a slower way to reach the same policy.

This is arithmetic a first-year student can do, and I still got it wrong repeatedly, for a structural reason worth naming: **sample-efficiency gains are legible and throughput costs are not.** A better learning curve shows up within an hour and feels like progress. A 40% throughput cost shows up at the end of the day as "I ran three experiments instead of five", which does not feel like anything at all.

<figure class="rep-fig" id="fig-breakeven">
  <p class="rep-fig-title">Figure 3 &nbsp;·&nbsp; The break-even plane</p>
  <p class="rep-fig-sub">drag the point, or load a real ablation</p>
  <div class="rep-widget">
    <div class="rep-scroll">
    <svg id="w-svg" viewBox="0 0 700 420" role="img"
         aria-label="A plane with throughput cost on the horizontal axis and sample-efficiency gain on the vertical axis. Points above the diagonal reduce time-to-target; points below increase it. The adv_top_frac ablation sits on the diagonal at 38 percent cost and 38 percent gain; the two opponent-mix points sit on the zero-gain axis at 44 and 53 percent cost.">
      <!-- plot area x 78..470, y 350..30 ; 4.9 px per % across, 4.0 px per % up -->
      <g class="grid">
        <line x1="78" y1="270" x2="470" y2="270"/>
        <line x1="78" y1="190" x2="470" y2="190"/>
        <line x1="78" y1="110" x2="470" y2="110"/>
        <line x1="176" y1="30" x2="176" y2="350"/>
        <line x1="274" y1="30" x2="274" y2="350"/>
        <line x1="372" y1="30" x2="372" y2="350"/>
      </g>
      <path d="M78 350 L470 30 L78 30 Z" fill="var(--wash-gain)"/>
      <path d="M78 350 L470 30 L470 350 Z" fill="var(--wash-cost)"/>
      <line x1="78" y1="350" x2="470" y2="30" stroke="var(--faint)" stroke-width="1.5" stroke-dasharray="5 4"/>
      <text x="92" y="56" fill="var(--blue)" style="font-size:11px">worth doing</text>
      <text x="92" y="71" class="mut">time-to-target falls</text>
      <text x="462" y="152" text-anchor="end" fill="var(--red)" style="font-size:11px">not worth doing</text>
      <text x="462" y="167" text-anchor="end" class="mut">time-to-target rises</text>
      <text x="196" y="248" class="mut" transform="rotate(-39.2 196 248)">break-even</text>
      <!-- axes -->
      <line class="axis" x1="78" y1="350" x2="470" y2="350"/>
      <line class="axis" x1="78" y1="30"  x2="78"  y2="350"/>
      <text x="78"  y="368" text-anchor="middle" class="mut">0</text>
      <text x="176" y="368" text-anchor="middle" class="mut">20</text>
      <text x="274" y="368" text-anchor="middle" class="mut">40</text>
      <text x="372" y="368" text-anchor="middle" class="mut">60</text>
      <text x="470" y="368" text-anchor="middle" class="mut">80%</text>
      <text x="274" y="392" text-anchor="middle" fill="var(--red)" style="font-size:11.5px">throughput cost  1 − R′/R</text>
      <text x="68" y="354" text-anchor="end" class="mut">0</text>
      <text x="68" y="274" text-anchor="end" class="mut">20</text>
      <text x="68" y="194" text-anchor="end" class="mut">40</text>
      <text x="68" y="114" text-anchor="end" class="mut">60</text>
      <text x="68" y="34"  text-anchor="end" class="mut">80%</text>
      <text x="26" y="190" text-anchor="middle" fill="var(--blue)" style="font-size:11.5px"
            transform="rotate(-90 26 190)">sample-efficiency gain  1 − N′/N</text>
      <!-- measured ablations -->
      <g id="w-marks">
        <circle cx="264.2" cy="198" r="5.5" fill="none" stroke="var(--ink)" stroke-width="1.8"/>
        <text x="277" y="195" fill="var(--ink)" style="font-size:11px">adv_top_frac 0.50</text>
        <text x="277" y="209" class="mut">−38% cost, −38% gain</text>
        <!-- leftmost point takes the higher leader so the two do not cross -->
        <circle cx="293.6" cy="350" r="5" fill="var(--red)"/>
        <polyline points="293.6,344 293.6,272 476,272" fill="none" stroke="var(--red)" stroke-width="1"/>
        <text x="484" y="268" fill="var(--red)" style="font-size:11px">20% frozen opponents</text>
        <text x="484" y="282" class="mut">−44% throughput</text>
        <circle cx="337.7" cy="350" r="5" fill="var(--red)"/>
        <polyline points="337.7,344 337.7,318 476,318" fill="none" stroke="var(--red)" stroke-width="1"/>
        <text x="484" y="314" fill="var(--red)" style="font-size:11px">50% transformer opponent</text>
        <text x="484" y="328" class="mut">−53% throughput</text>
        <text x="484" y="200" class="mut">gain unmeasured for both —</text>
        <text x="484" y="214" class="mut">they are plotted at zero</text>
      </g>
      <!-- draggable point -->
      <g id="w-drag" style="cursor:grab">
        <line id="w-vline" x1="264.2" y1="350" x2="264.2" y2="198" stroke="var(--faint)" stroke-width="1" stroke-dasharray="3 3"/>
        <line id="w-hline" x1="78" y1="198" x2="264.2" y2="198" stroke="var(--faint)" stroke-width="1" stroke-dasharray="3 3"/>
        <circle id="w-pt" cx="264.2" cy="198" r="9" fill="var(--blue)" stroke="var(--paper)" stroke-width="2.5"/>
      </g>
      <rect id="w-hit" x="78" y="30" width="392" height="320" fill="transparent" style="cursor:crosshair"/>
    </svg>
    </div>
    <div class="rep-wpanel">
      <div class="rep-readout">
        <span class="rk">time to the same strength</span>
        <span class="rv" id="w-big">1.00×</span>
        <div class="rep-rbar"><i id="w-bar"></i><span class="mid"></span></div>
        <span class="rn" id="w-note">exactly break-even</span>
      </div>
      <div class="rep-ctl">
        <label>sample-efficiency gain <b><span id="w-gv">38</span>%</b></label>
        <input type="range" id="w-g" min="0" max="80" value="38" step="1" aria-label="sample-efficiency gain, percent">
        <label>throughput cost <b><span id="w-cv">38</span>%</b></label>
        <input type="range" id="w-c" class="clock" min="0" max="80" value="38" step="1" aria-label="throughput cost, percent">
      </div>
      <div class="rep-presets">
        <span class="ph">measured</span>
        <button type="button" data-g="38" data-c="38">adv_top_frac 0.50</button>
        <button type="button" data-g="0" data-c="44">20% frozen opponents</button>
        <button type="button" data-g="0" data-c="53">transformer opponent</button>
        <button type="button" data-g="0" data-c="-21">FP8 (+21% throughput)</button>
      </div>
    </div>
  </div>
  <figcaption><span class="fig-number">Figure 3</span>An intervention is worth taking only if it sits above
    the diagonal. Three of the four measured points are on or below it. <code>adv_top_frac 0.50</code>
    landed on the line to two significant figures and duly produced a statistical tie in 2,048 games
    (§7). The two opponent-mix points have a measured throughput cost and an <em>unmeasured</em>
    sample-efficiency gain — which is precisely the asymmetry that makes this mistake easy to
    make. FP8 sits off the plane to the left: it bought throughput and destroyed the policy (§9).</figcaption>
</figure>

## The 11.7× I did not take

The training stack is a JAX/Equinox PPO trainer built on a C and CUDA game engine from PufferLib. The policy, `cnn_global`, is a 2,724,746-parameter residual CNN: a 45-channel observation at 21 × 21 (31 base channels plus history stacks, including scouting and deathtouch planes), a 1 × 1 stem, six 192-wide 3 × 3 residual blocks with a global-context vector injected into each, a 4,410-way masked policy head laid out as ten action planes over the board, and a 128-bin HL-Gauss value head. Trained with Muon throughout.

Early on I measured the two stacks against each other on one A100-40GB, and the result was not close. PufferLib's native CUDA path ran **391,412 agent-steps per second**; my JAX trainer ran **33,512**. A factor of 11.7. The whole measurement cost twenty-five cents, and for about a day I treated switching as obviously correct.

It was not, and finding out why produced the single most useful process change of the week. When I audited what the native backend had actually instantiated, it was not the CNN. The custom CUDA Generals CNN was shape-gated for an older 18-channel, 4,609-action layout; the competition environment presents 19 channels and 5,185 actions, so the gate silently fell through to a generic flat encoder with a MinGRU — 4,325,888 parameters, a completely different model. It was 11.7× faster because it was doing something much easier.

I trained it anyway, to be sure. It reached 100,139,008 agent-steps in six minutes and thirty-five seconds at 252.5k SPS, entropy fell from 8.55 to 2.08, and every training counter moved. Then I evaluated it across stacks:

<figure class="rep-fig">
  <p class="rep-fig-title">Figure 4 &nbsp;·&nbsp; Throughput of a model you cannot use</p>
  <div class="rep-scroll">
  <svg viewBox="0 0 1080 260" role="img"
       aria-label="The native CUDA stack ran 391,412 agent-steps per second against the JAX stack's 33,512, an 11.7 times advantage. But the fast stack had silently selected a different, weaker model, which won zero of 448 games against scripted opponents.">
    <!-- left panel: SPS -->
    <text class="ttl" x="0" y="14">TRAINING THROUGHPUT · one A100-SXM4-40GB, identical box</text>
    <g transform="translate(0,32)">
      <text x="0" y="14" style="font-size:11.5px" fill="var(--ink)">PufferLib native CUDA</text>
      <text x="0" y="30" class="mut">flat encoder + MinGRU · 4,325,888 params</text>
      <rect x="200" y="0" width="300" height="30" rx="2" fill="var(--red)"/>
      <text x="510" y="20" fill="var(--red)" style="font-size:13px">391,412 SPS</text>
      <text x="0" y="76" style="font-size:11.5px" fill="var(--ink)">JAX / Equinox</text>
      <text x="0" y="92" class="mut">history transformer · 5,128,274 params</text>
      <rect x="200" y="62" width="25.7" height="30" rx="2" fill="var(--blue)"/>
      <text x="236" y="82" fill="var(--blue)" style="font-size:13px">33,512 SPS</text>
      <text x="330" y="82" class="mut">11.7× slower</text>
    </g>
    <!-- divider -->
    <line x1="620" y1="0" x2="620" y2="240" stroke="var(--line)"/>
    <!-- right panel: strength of the fast model -->
    <text class="ttl" x="648" y="14">WHAT THE FAST MODEL LEARNED · after 100 M steps</text>
    <g transform="translate(648,34)">
      <text x="0" y="12" style="font-size:11.5px" fill="var(--ink)">vs 7 scripted bots · 448 games</text>
      <rect x="0" y="22" width="0.0" height="26" fill="var(--blue)"/>
      <rect x="0" y="22" width="26.6" height="26" fill="var(--red)"/>
      <rect x="28.6" y="22" width="383.4" height="26" fill="url(#hatch)" stroke="var(--line)"/>
      <text x="0" y="66" class="mut">0 wins</text>
      <text x="76" y="66" fill="var(--red)" style="font-size:11px">29 losses</text>
      <text x="412" y="40" text-anchor="end" fill="var(--muted)" style="font-size:11px">419 draws</text>
      <text x="0" y="104" style="font-size:11.5px" fill="var(--ink)">vs my champion · 128 games</text>
      <rect x="0" y="114" width="412" height="26" fill="var(--red)"/>
      <text x="8" y="132" fill="var(--paper)" style="font-size:11px">0 W  ·  112 L  ·  16 D</text>
      <text x="0" y="172" class="mut">mean game length 1,145–1,200 turns — it had learned to stall</text>
    </g>
  </svg>
  </div>
  <figcaption><span class="fig-number">Figure 4</span>The 11.7× throughput advantage was real and the model
    behind it was worthless: no scripted opponent conceded a single win in 448 games, and games ran
    to the 1,200-turn truncation. Throughput is only a denominator when the numerator is the model
    you intend to ship.</figcaption>
</figure>

Out of that came a rule I now apply before renting anything, and which I would defend as a general practice in RL engineering: **the runtime architecture identity gate**. Before a run is allowed to spend money, it must log and assert, from the live objects rather than from the configuration file, the resolved encoder and decoder implementations, the observation shape, the action shape, the parameter count, the recurrent-state shape, and a fingerprint of the parameter layout. A config name is a request. A backend is free to ignore it and frequently does.

> A learning smoke test proves that the resolved model learns. It never proves that the model you asked for was the one selected.

## Two bugs worth more than any architecture change

The two largest single improvements of the whole week were both corrections to things I had already implemented and believed were working. Neither was a new idea. Both had the same shape: the arithmetic was locally defensible and the trajectory distribution was wrong.

### 4.1 · The draw penalty that never reached the loss

Draws are the failure mode of this ruleset. A policy that accumulates a large economy but cannot walk into an enemy general before turn 1200 scores nothing, and on a competition ladder that is indistinguishable from being bad. So I added a penalty on truncation: a draw is worth −0.5, later −1.5 in the shaped return. Draw rates did not move. I assumed the coefficient was too small and turned it up. They still did not move.

The penalty was never reaching the gradient. Truncation was being handled as a *reset* rather than as a terminal, in three separate places that were each individually reasonable:

1. the value bootstrap read \\(V(s_{t+1})\\) from the state *after* the environment had reset, i.e. from a fresh board;
2. the GAE carry was cut at the episode boundary, so nothing propagated backwards past it;
3. the final row of the rollout — the only row carrying the −1.5 target — was masked out of the loss as an incomplete transition.

Each of those is a thing you do somewhere in a PPO implementation. Together they made the penalty exactly invisible: the one row that contained the signal was the one row excluded from the update. The fix is to treat truncation as a true terminal in the advantage computation,

<div class="formula">\(\delta_t = r_t + \gamma(1-d_t)V(s_{t+1}) - V(s_t), \qquad \hat{A}_t = \delta_t + \gamma\lambda(1-d_t)\hat{A}_{t+1}\)<small>with \(d_t = 1\) on a 1200-turn truncation, and the row restored to the batch</small></div>

<figure class="rep-fig">
  <p class="rep-fig-title">Figure 5 &nbsp;·&nbsp; Where the draw penalty went</p>
  <div class="rep-scroll">
  <svg viewBox="0 0 1080 440" role="img"
       aria-label="Two stacked views of the same rollout buffer at the 1200-turn truncation. In the before version the value bootstrap reads the post-reset board, the advantage carry is cut at the boundary, and the row carrying the minus 1.5 draw target is masked out of the loss, so no draw term reaches the gradient. In the after version truncation is marked terminal, no bootstrap is taken, the penalty propagates back through GAE, and the row stays in the batch.">
    <defs>
      <marker id="ar2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0 0 L10 5 L0 10 z" fill="currentColor"/>
      </marker>
    </defs>
    <!-- ============ BEFORE ============ -->
    <g color="var(--faint)">
      <text class="ttl" x="0" y="12" fill="var(--red)">BEFORE — the penalty never reaches the gradient</text>
      <!-- rollout strip -->
      <text x="245" y="36" text-anchor="middle" class="mut">turn 1200</text>
      <text x="328" y="36" text-anchor="middle" class="mut">reset</text>
      <rect x="40"  y="44" width="52" height="34" fill="var(--paper-2)" stroke="var(--line)"/>
      <rect x="94"  y="44" width="52" height="34" fill="var(--paper-2)" stroke="var(--line)"/>
      <rect x="148" y="44" width="52" height="34" fill="var(--paper-2)" stroke="var(--line)"/>
      <text x="118" y="66" text-anchor="middle" class="mut">t · · ·</text>
      <rect x="202" y="44" width="86" height="34" fill="var(--wash-cost-2)"
            stroke="var(--red)" stroke-dasharray="4 3"/>
      <text x="245" y="66" text-anchor="middle" fill="var(--red)" style="font-size:11px">r = −1.5</text>
      <line x1="295" y1="30" x2="295" y2="94" stroke="var(--faint)" stroke-width="1.5" stroke-dasharray="3 3"/>
      <rect x="302" y="44" width="52" height="34" fill="var(--paper-2)" stroke="var(--line)"/>
      <text x="328" y="66" text-anchor="middle" class="mut">new</text>
      <!-- fault 1: bootstrap crosses the reset -->
      <path d="M245 82 L245 100 L328 100 L328 82" fill="none" stroke="var(--red)" stroke-width="1.5"
            marker-end="url(#ar2)" color="var(--red)"/>
      <text x="400" y="104" fill="var(--red)" style="font-size:11px">1 · bootstraps V from the post-reset board</text>
      <!-- fault 2: carry cut -->
      <line x1="202" y1="132" x2="150" y2="132" stroke="var(--red)" stroke-width="1.5" stroke-dasharray="4 3"/>
      <line x1="184" y1="123" x2="166" y2="141" stroke="var(--red)" stroke-width="2"/>
      <line x1="166" y1="123" x2="184" y2="141" stroke="var(--red)" stroke-width="2"/>
      <text x="400" y="136" fill="var(--red)" style="font-size:11px">2 · advantage carry cut at the boundary</text>
      <!-- fault 3: masked -->
      <rect x="202" y="152" width="86" height="26" fill="url(#hatch)" stroke="var(--red)"/>
      <text x="245" y="169" text-anchor="middle" fill="var(--red)" style="font-size:10px">MASKED</text>
      <text x="400" y="169" fill="var(--red)" style="font-size:11px">3 · the only row with the target is dropped</text>
      <text x="0" y="200" fill="var(--red)" style="font-size:12px">net effect: ∂L/∂θ contains no draw term at all</text>
      <line x1="720" y1="0" x2="720" y2="206" stroke="var(--line)"/>
      <text x="744" y="60" style="font-size:11px">Each of the three is a</text>
      <text x="744" y="78" style="font-size:11px">reasonable thing to do</text>
      <text x="744" y="96" style="font-size:11px">somewhere in a PPO loop.</text>
      <text x="744" y="130" style="font-size:11px" fill="var(--red)">Together they delete the</text>
      <text x="744" y="148" style="font-size:11px" fill="var(--red)">reward term exactly.</text>
    </g>
    <line x1="0" y1="228" x2="1080" y2="228" stroke="var(--line)"/>
    <!-- ============ AFTER ============ -->
    <g transform="translate(0,252)" color="var(--faint)">
      <text class="ttl" x="0" y="12" fill="var(--blue)">AFTER — truncation is a true terminal</text>
      <text x="245" y="36" text-anchor="middle" class="mut">turn 1200</text>
      <text x="328" y="36" text-anchor="middle" class="mut">reset</text>
      <rect x="40"  y="44" width="52" height="34" fill="var(--paper-2)" stroke="var(--line)"/>
      <rect x="94"  y="44" width="52" height="34" fill="var(--paper-2)" stroke="var(--line)"/>
      <rect x="148" y="44" width="52" height="34" fill="var(--paper-2)" stroke="var(--line)"/>
      <text x="118" y="66" text-anchor="middle" class="mut">t · · ·</text>
      <rect x="202" y="44" width="86" height="34" fill="var(--wash-gain)" stroke="var(--blue)" stroke-width="1.5"/>
      <text x="245" y="66" text-anchor="middle" fill="var(--blue)" style="font-size:11px">r = −1.5</text>
      <line x1="295" y1="30" x2="295" y2="94" stroke="var(--faint)" stroke-width="1.5" stroke-dasharray="3 3"/>
      <rect x="302" y="44" width="52" height="34" fill="var(--paper-2)" stroke="var(--line)"/>
      <text x="328" y="66" text-anchor="middle" class="mut">new</text>
      <!-- 1: no bootstrap -->
      <path d="M245 82 L245 100 L318 100" fill="none" stroke="var(--blue)" stroke-width="1.5"/>
      <line x1="330" y1="91" x2="312" y2="109" stroke="var(--blue)" stroke-width="2"/>
      <line x1="312" y1="91" x2="330" y2="109" stroke="var(--blue)" stroke-width="2"/>
      <text x="400" y="104" fill="var(--blue)" style="font-size:11px">1 · d = 1, so (1−d)·V(s′) = 0 — no bootstrap</text>
      <!-- 2: carry propagates -->
      <line x1="202" y1="132" x2="48" y2="132" stroke="var(--blue)" stroke-width="1.5"
            marker-end="url(#ar2)" color="var(--blue)"/>
      <text x="400" y="136" fill="var(--blue)" style="font-size:11px">2 · the −1.5 propagates back through GAE</text>
      <!-- 3: row kept -->
      <rect x="202" y="152" width="86" height="26" fill="var(--wash-gain)" stroke="var(--blue)"/>
      <text x="245" y="169" text-anchor="middle" fill="var(--blue)" style="font-size:10px">KEPT</text>
      <text x="400" y="169" fill="var(--blue)" style="font-size:11px">3 · the row stays in the loss</text>
      <text x="0" y="200" fill="var(--blue)" style="font-size:12px">held stable to iteration 1,700 · scripted 97–100% W · self-play draws 6–9%</text>
      <line x1="720" y1="0" x2="720" y2="206" stroke="var(--line)"/>
      <text x="744" y="60" style="font-size:11px">Draw rate responded to</text>
      <text x="744" y="78" style="font-size:11px">the coefficient for the</text>
      <text x="744" y="96" style="font-size:11px">first time.</text>
      <text x="744" y="130" style="font-size:11px" fill="var(--blue)">One-line change; the</text>
      <text x="744" y="148" style="font-size:11px" fill="var(--blue)">largest single gain</text>
      <text x="744" y="166" style="font-size:11px" fill="var(--blue)">of the week.</text>
    </g>
  </svg>
  </div>
  <figcaption><span class="fig-number">Figure 5</span>The three defensible decisions that jointly deleted a
    reward term. The fixed run stayed healthy through 1,700 iterations. It also exonerated a feature
    I was about to remove: the no-forced-build control had collapsed <em>before</em> the fix, so the
    collapse I had blamed on forced-build exploration was the missing draw penalty all along.</figcaption>
</figure>

<div class="callout warning">
  <div class="callout-title">THE LESSON</div>
  <p>A reward you cannot find in the gradient is a comment. I now trace the full path — config, rollout, return, loss, metric — before changing a coefficient, because turning up a number that reaches nothing produces a very convincing null result.</p>
</div>

### 4.2 · Symmetry augmentation applied one step too late

A Generals board has the eight symmetries of the square: four rotations and four reflections. That is free data augmentation, and the first implementation looked like this: collect a transition in the canonical orientation; inside each PPO epoch, draw a random \\(g \in D_4\\) and transform the stored observation, action and mask; then recompute the behaviour log-probability \\(\log \pi_{\theta_{\text{old}}}(g \cdot a \mid g \cdot s)\\) in the transformed frame.

That last step is the one that looks careful and is the reason it is wrong. PPO's importance ratio

<div class="formula">\(\rho_t(\theta) = \dfrac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}\)</div>

is only an importance weight if \\(a_t\\) was *sampled* from \\(\pi_{\theta_{\text{old}}}\\) in state \\(s_t\\). The transformed action \\(g \cdot a_t\\) was never sampled by a policy looking at \\(g \cdot s_t\\); it was sampled by a policy looking at \\(s_t\\) and then rotated. Recomputing the denominator makes the arithmetic self-consistent and the estimator biased. As a bonus, a given sample could receive a different frame on each PPO pass, which adds gradient variance for nothing.

<figure class="rep-fig">
  <p class="rep-fig-title">Figure 6 &nbsp;·&nbsp; Where the group action belongs</p>
  <div class="rep-scroll">
  <svg viewBox="0 0 1080 345" role="img"
       aria-label="Two pipelines. In the broken version the D4 transform is applied to stored samples during the PPO update and the behaviour log-probability is recomputed, which breaks the on-policy identity. In the corrected version a frame is drawn once per episode before inference, so the behaviour policy samples inside the transformed frame and only the environment-facing action is inverse-transformed.">
    <defs>
      <marker id="ar3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0 0 L10 5 L0 10 z" fill="currentColor"/>
      </marker>
    </defs>
    <!-- BROKEN -->
    <text class="ttl" x="0" y="12" fill="var(--red)">BEFORE — g applied to stored samples</text>
    <g transform="translate(0,32)" color="var(--faint)">
      <rect x="0" y="0" width="92" height="36" fill="var(--paper-2)" stroke="var(--line)"/>
      <text x="46" y="22" text-anchor="middle">env</text>
      <line x1="92" y1="18" x2="126" y2="18" stroke="currentColor" marker-end="url(#ar3)"/>
      <rect x="128" y="0" width="120" height="36" fill="var(--paper-2)" stroke="var(--line)"/>
      <text x="188" y="22" text-anchor="middle">π samples a | s</text>
      <line x1="248" y1="18" x2="282" y2="18" stroke="currentColor" marker-end="url(#ar3)"/>
      <rect x="284" y="0" width="104" height="36" fill="var(--paper-2)" stroke="var(--line)"/>
      <text x="336" y="22" text-anchor="middle">buffer</text>
      <!-- the misplaced transform -->
      <line x1="388" y1="18" x2="422" y2="18" stroke="currentColor" marker-end="url(#ar3)"/>
      <rect x="424" y="-14" width="152" height="64" fill="var(--wash-cost-2)" stroke="var(--red)" stroke-width="1.5"/>
      <text x="500" y="8" text-anchor="middle" fill="var(--red)" style="font-size:11px">PPO epoch</text>
      <text x="500" y="26" text-anchor="middle" fill="var(--red)" style="font-size:11px">apply g · recompute</text>
      <text x="500" y="42" text-anchor="middle" fill="var(--red)" style="font-size:11px">log π_old(g·a | g·s)</text>
      <text x="0" y="82" fill="var(--red)" style="font-size:11.5px">g·a was never sampled in state g·s — the ratio stops being an importance weight</text>
      <text x="0" y="100" class="mut">and a sample can draw a different frame on every pass</text>
    </g>
    <!-- AFTER -->
    <text class="ttl" x="0" y="196" fill="var(--blue)">AFTER — g drawn per episode, before inference</text>
    <g transform="translate(0,256)" color="var(--faint)">
      <rect x="0" y="0" width="92" height="36" fill="var(--paper-2)" stroke="var(--line)"/>
      <text x="46" y="22" text-anchor="middle">env</text>
      <line x1="92" y1="18" x2="126" y2="18" stroke="currentColor" marker-end="url(#ar3)"/>
      <rect x="128" y="-14" width="120" height="64" fill="var(--wash-gain)" stroke="var(--blue)" stroke-width="1.5"/>
      <text x="188" y="6" text-anchor="middle" fill="var(--blue)" style="font-size:11px">apply g to</text>
      <text x="188" y="24" text-anchor="middle" fill="var(--blue)" style="font-size:11px">obs + legal mask</text>
      <text x="188" y="42" text-anchor="middle" fill="var(--blue)" style="font-size:10px">one g per episode</text>
      <line x1="248" y1="18" x2="282" y2="18" stroke="currentColor" marker-end="url(#ar3)"/>
      <rect x="284" y="0" width="150" height="36" fill="var(--paper-2)" stroke="var(--line)"/>
      <text x="359" y="22" text-anchor="middle">π samples a′ | g·s</text>
      <line x1="434" y1="18" x2="468" y2="18" stroke="currentColor" marker-end="url(#ar3)"/>
      <rect x="470" y="0" width="106" height="36" fill="var(--paper-2)" stroke="var(--line)"/>
      <text x="523" y="22" text-anchor="middle">buffer</text>
      <text x="523" y="52" text-anchor="middle" fill="var(--blue)" style="font-size:10px">stored in frame g</text>
      <!-- inverse only to env -->
      <path d="M359 -14 L359 -30 L46 -30 L46 -14" fill="none" stroke="var(--blue)" stroke-width="1.5"
            marker-end="url(#ar3)" color="var(--blue)"/>
      <text x="200" y="-36" text-anchor="middle" fill="var(--blue)" style="font-size:10.5px">g⁻¹ applied only to the action sent to env.step</text>
    </g>
    <!-- results -->
    <line x1="620" y1="0" x2="620" y2="325" stroke="var(--line)"/>
    <text class="ttl" x="648" y="12">RESULT · 2,048 seat-balanced games each</text>
    <g transform="translate(648,32)">
      <text x="0" y="12" style="font-size:11.5px" fill="var(--ink)">corrected D4  vs  no D4</text>
      <text x="0" y="30" class="rec" style="font-size:12px" fill="var(--blue)">1083 – 949 – 7</text>
      <text x="150" y="30" fill="var(--blue)" style="font-size:12px">margin +134</text>
      <text x="270" y="30" class="mut">bar ≈ 74</text>
      <text x="0" y="68" style="font-size:11.5px" fill="var(--ink)">corrected D4  vs  frozen parent</text>
      <text x="0" y="86" class="rec" style="font-size:12px">1180 – 854 – 3</text>
      <text x="150" y="86" style="font-size:12px">+326</text>
      <text x="0" y="124" style="font-size:11.5px" fill="var(--ink)">no D4  vs  frozen parent</text>
      <text x="0" y="142" class="rec" style="font-size:12px">1076 – 966 – 1</text>
      <text x="150" y="142" style="font-size:12px">+110</text>
      <text x="0" y="182" class="mut">the control also beat the parent —</text>
      <text x="0" y="198" class="mut">which is why only the first row is evidence</text>
    </g>
  </svg>
  </div>
  <figcaption><span class="fig-number">Figure 6</span>The correction moves one operation from the update
    into the rollout. Drawing the frame before inference restores PPO's on-policy identity and gives
    the policy a temporally consistent coordinate system for the whole episode. Note the third row:
    the no-augmentation control gained +110 on its parent from continued training alone, which is
    the confound that §6 is about.</figcaption>
</figure>

> In on-policy RL a symmetry transform has to preserve the entire trajectory distribution, not merely the tensor shapes.

## How I decided anything

Self-play win rate against yourself is always 50%, so every decision has to come from outside the run. Candidates were compared in seat-balanced games against a fixed opponent, and promoted only when

<div class="formula">\(W - L \geq 1.65\sqrt{W+L}\)<small>the normal approximation to a fair-coin binomial over the decisive games; 1.65 is the one-sided 95% point</small></div>

Under the null the standard deviation of \\(W-L\\) is \\(\sqrt{W+L}\\). Draws are excluded from the test and reported separately, because in this ruleset a draw is a distinct failure mode rather than half a win.

Two things about that gate cost me real time to learn.

**First, small evaluations are not cheap, they are free of information.** I spent the first several days making calls from a 32-game greedy match against a random mover, because it ran in twenty seconds. The same checkpoint scored 97, 97 and 88 on three different boxes. At \\(n=32\\) the gate cannot resolve anything smaller than about 13 percentage points, which is larger than every effect I was trying to measure. It is a smoke test, and I had been reading it as a result.

**Second, the eval scales sublinearly, so the cheap option was also the imprecise one for no reason.** A fixed 50–75 second compile dominates the small sizes:

<figure class="rep-fig">
  <p class="rep-fig-title">Figure 7 &nbsp;·&nbsp; Why the gate runs at 2,048 games</p>
  <div class="rep-scroll">
  <svg viewBox="0 0 1080 250" role="img"
       aria-label="Two panels. Quadrupling the evaluation from 512 to 2048 games raises the margin over the significance bar from 3.6 times to 9.5 times, while wall-clock time rises only from 158 to 338 seconds because a fixed compile dominates.">
    <!-- panel A: resolving power -->
    <text class="ttl" x="0" y="12">RESOLVING POWER &#183; margin as a multiple of the significance bar</text>
    <g transform="translate(0,30)">
      <line class="axis" x1="90" y1="150" x2="470" y2="150"/>
      <rect x="120" y="96"  width="58" height="54"  fill="var(--blue)" rx="2"/>
      <rect x="240" y="66"  width="58" height="84"  fill="var(--blue)" rx="2"/>
      <rect x="360" y="8"   width="58" height="142" fill="var(--blue)" rx="2"/>
      <text x="149" y="88"  text-anchor="middle" fill="var(--blue)" style="font-size:12.5px">3.6&#215;</text>
      <text x="269" y="58"  text-anchor="middle" fill="var(--blue)" style="font-size:12.5px">6.0&#215;</text>
      <text x="389" y="0"   text-anchor="middle" fill="var(--blue)" style="font-size:12.5px">9.5&#215;</text>
      <text x="149" y="168" text-anchor="middle" class="mut">512</text>
      <text x="269" y="168" text-anchor="middle" class="mut">1024</text>
      <text x="389" y="168" text-anchor="middle" class="mut">2048</text>
      <text x="80"  y="168" text-anchor="end" class="mut">games</text>
      <text x="90"  y="192" class="mut">measured on one &#8776;68%-win matchup</text>
    </g>
    <line x1="540" y1="0" x2="540" y2="230" stroke="var(--line)"/>
    <!-- panel B: wall clock -->
    <text class="ttl" x="570" y="12">WALL CLOCK &#183; seconds, WSL2 RTX 4070</text>
    <g transform="translate(570,30)">
      <line class="axis" x1="90" y1="150" x2="470" y2="150"/>
      <rect x="120" y="84"  width="58" height="66"  fill="var(--red)" rx="2"/>
      <rect x="240" y="55"  width="58" height="95"  fill="var(--red)" rx="2"/>
      <rect x="360" y="9"   width="58" height="141" fill="var(--red)" rx="2"/>
      <rect x="120" y="124" width="58" height="26" fill="url(#hatch)" stroke="var(--red)" stroke-width=".8"/>
      <rect x="240" y="124" width="58" height="26" fill="url(#hatch)" stroke="var(--red)" stroke-width=".8"/>
      <rect x="360" y="124" width="58" height="26" fill="url(#hatch)" stroke="var(--red)" stroke-width=".8"/>
      <text x="149" y="76"  text-anchor="middle" fill="var(--red)" style="font-size:12.5px">158 s</text>
      <text x="269" y="47"  text-anchor="middle" fill="var(--red)" style="font-size:12.5px">228 s</text>
      <text x="389" y="1"   text-anchor="middle" fill="var(--red)" style="font-size:12.5px">338 s</text>
      <text x="149" y="168" text-anchor="middle" class="mut">512</text>
      <text x="269" y="168" text-anchor="middle" class="mut">1024</text>
      <text x="389" y="168" text-anchor="middle" class="mut">2048</text>
      <text x="90"  y="192" class="mut">hatched: the fixed 50&#8211;75 s compile, paid at every size</text>
    </g>
  </svg>
  </div>
  <figcaption><span class="fig-number">Figure 7</span>Four times the games costs 2.1× the wall clock and
    halves the error bar. Once the compile is amortised there is no reason to run the imprecise
    version — which meant that for several days I had been buying uncertainty at nearly full
    price.</figcaption>
</figure>

## The ladder, and what it does not prove

With roughly 48 hours left I stopped redesigning the network. The 2.72 M CNN was frozen, training was pure self-play with Muon, and each experiment was a matched two-hour arm on 2× RTX 5090. Every candidate then played its sibling *and* its frozen parent at up to 2,048 seat-balanced games. Six generations were promoted:

<figure class="rep-fig">
  <p class="rep-fig-title">Figure 8 &nbsp;·&nbsp; Six promotions, two causal claims</p>
  <div class="rep-legend">
    <span><i style="background:var(--blue)"></i>treatment vs frozen parent</span>
    <span><i style="background:var(--red)"></i>control vs frozen parent</span>
    <span><i style="background:var(--ink);border-radius:50%"></i>treatment vs control — the causal statistic</span>
    <span><i style="background:transparent;border-left:2px dashed var(--faint);width:2px;border-radius:0"></i>promotion bar, ≈74</span>
  </div>
  <div class="rep-scroll">
  <svg viewBox="0 0 1080 340" role="img"
       aria-label="A dumbbell chart of six generations. In each, both the treatment and the control are measured against the frozen parent, and the direct treatment-versus-control margin is marked separately. In three of five generations with a recorded control, the control also cleared the promotion bar, so beating the parent does not establish that the change caused the improvement.">
    <!-- x scale: -60 .. 340 margin  ->  x = 300 + margin * 1.85  ; 0 at x=300 -->
    <g class="grid">
      <line x1="300" y1="26" x2="300" y2="300"/>
      <line x1="485" y1="26" x2="485" y2="300"/>
      <line x1="670" y1="26" x2="670" y2="300"/>
      <line x1="855" y1="26" x2="855" y2="300"/>
    </g>
    <text x="300" y="318" text-anchor="middle" class="mut">0</text>
    <text x="485" y="318" text-anchor="middle" class="mut">+100</text>
    <text x="670" y="318" text-anchor="middle" class="mut">+200</text>
    <text x="855" y="318" text-anchor="middle" class="mut">+300</text>
    <text x="670" y="336" text-anchor="middle" class="mut">margin  W − L  (2,048 games)</text>
    <!-- promotion bar at +74 -> x = 300 + 74*1.85 = 436.9 -->
    <line x1="436.9" y1="20" x2="436.9" y2="300" stroke="var(--faint)" stroke-width="1.5" stroke-dasharray="5 4"/>
    <text x="436.9" y="14" text-anchor="middle" class="mut">promotion bar</text>
    <!-- rows -->
    <!-- g03 -->
    <g transform="translate(0,52)">
      <text x="0" y="4" style="font-size:12px" fill="var(--ink)">g03</text>
      <text x="42" y="4" class="mut">corrected D4 augmentation</text>
      <line x1="503.5" y1="0" x2="903.1" y2="0" stroke="var(--line)" stroke-width="3"/>
      <circle cx="903.1" cy="0" r="6" fill="var(--blue)"/>
      <circle cx="503.5" cy="0" r="6" fill="var(--red)"/>
      <polygon points="547.9,-7 555.9,0 547.9,7 539.9,0" fill="var(--ink)"/>
      <text x="547.9" y="-14" text-anchor="middle" fill="var(--ink)" style="font-size:11px">+134</text>
    </g>
    <!-- g04 -->
    <g transform="translate(0,96)">
      <text x="0" y="4" style="font-size:12px" fill="var(--ink)">g04</text>
      <text x="42" y="4" class="mut">minibatch 1024 over 2048</text>
      <text x="503.5" y="4" class="mut">promoted; no direct treatment-vs-control record survives</text>
    </g>
    <!-- g05 -->
    <g transform="translate(0,140)">
      <text x="0" y="4" style="font-size:12px" fill="var(--ink)">g05</text>
      <text x="42" y="4" class="mut">two PPO passes</text>
      <line x1="348.1" y1="0" x2="518.3" y2="0" stroke="var(--line)" stroke-width="3"/>
      <circle cx="518.3" cy="0" r="6" fill="var(--blue)"/>
      <circle cx="348.1" cy="0" r="6" fill="var(--red)"/>
      <polygon points="366.6,-7 374.6,0 366.6,7 358.6,0" fill="var(--ink)"/>
      <text x="366.6" y="-14" text-anchor="middle" fill="var(--ink)" style="font-size:11px">+36</text>
    </g>
    <!-- g06 -->
    <g transform="translate(0,184)">
      <text x="0" y="4" style="font-size:12px" fill="var(--ink)">g06</text>
      <text x="42" y="4" class="mut">GAE λ = 0.9 over 0.7</text>
      <line x1="497.9" y1="0" x2="757.0" y2="0" stroke="var(--line)" stroke-width="3"/>
      <circle cx="757.0" cy="0" r="6" fill="var(--blue)"/>
      <circle cx="497.9" cy="0" r="6" fill="var(--red)"/>
      <polygon points="618.2,-7 626.2,0 618.2,7 610.2,0" fill="var(--ink)"/>
      <text x="618.2" y="-14" text-anchor="middle" fill="var(--ink)" style="font-size:11px">+172</text>
    </g>
    <!-- g07 -->
    <g transform="translate(0,228)">
      <text x="0" y="4" style="font-size:12px" fill="var(--ink)">g07</text>
      <text x="42" y="4" class="mut">entropy floor 0.003 over 0.001</text>
      <line x1="331.5" y1="0" x2="557.2" y2="0" stroke="var(--line)" stroke-width="3"/>
      <circle cx="557.2" cy="0" r="6" fill="var(--blue)"/>
      <circle cx="331.5" cy="0" r="6" fill="var(--red)"/>
      <polygon points="445.6,-7 453.6,0 445.6,7 437.6,0" fill="var(--ink)"/>
      <text x="445.6" y="-14" text-anchor="middle" fill="var(--ink)" style="font-size:11px">+83</text>
    </g>
    <!-- g08 -->
    <g transform="translate(0,272)">
      <text x="0" y="4" style="font-size:12px" fill="var(--ink)">g08</text>
      <text x="42" y="4" class="mut">advantage top-50% over top-25%</text>
      <line x1="448.0" y1="0" x2="505.4" y2="0" stroke="var(--line)" stroke-width="3"/>
      <circle cx="505.4" cy="0" r="6" fill="var(--blue)"/>
      <circle cx="448.0" cy="0" r="6" fill="var(--red)"/>
      <polygon points="261.2,-7 269.2,0 261.2,7 253.2,0" fill="var(--ink)"/>
      <text x="261.2" y="-14" text-anchor="middle" fill="var(--red)" style="font-size:11px">−21</text>
    </g>
  </svg>
  </div>
  <figcaption><span class="fig-number">Figure 8</span>Every promoted arm cleared the bar against its frozen
    parent — and in three of the five generations with a recorded control, <em>so did the control</em>
    (g03, g06, g08). Beating your parent certifies the checkpoint; it does not identify the cause.
    Read down the diamonds instead: only g06 and g03 have a direct treatment-vs-control margin that
    clears the bar on its own.</figcaption>
</figure>

Six promotions look like six successful ablations. They are not, and Figure 8 is the reason. In three of the five generations where I ran a matched control, the control also beat the frozen parent by a significant margin. Two more hours of self-play is itself a powerful treatment — the lineage had not plateaued — so beating the parent proves that the new checkpoint is better and says nothing about *why*. The only statistic that isolates the change is the direct treatment-versus-control match, and those margins are much smaller:

<div class="table-wrap">
<table>
  <thead><tr><th>Change</th><th>Direct W–L–D</th><th>Margin</th><th>Evidence</th></tr></thead>
  <tbody>
    <tr><td><b>GAE λ 0.9 over 0.7</b></td><td class="rep-rec">1102 – 930 – 13</td><td class="rep-num pos">+172</td><td><span class="rep-chip ok">strong</span></td></tr>
    <tr><td><b>Corrected D4 augmentation</b></td><td class="rep-rec">1083 – 949 – 7</td><td class="rep-num pos">+134</td><td><span class="rep-chip ok">strong</span></td></tr>
    <tr><td><b>Entropy floor 0.003</b></td><td class="rep-rec">1055 – 972 – 14</td><td class="rep-num pos">+83</td><td><span class="rep-chip ok">moderate</span></td></tr>
    <tr><td><b>Two PPO passes</b></td><td class="rep-rec">1033 – 997 – 7</td><td class="rep-num">+36</td><td><span class="rep-chip mid">tie — promoted on the parent gate</span></td></tr>
    <tr><td><b>Advantage top-50%</b></td><td class="rep-rec">1009 – 1030 – 2</td><td class="rep-num neg">−21</td><td><span class="rep-chip mid">tie — promoted on a tie-break</span></td></tr>
    <tr><td><b>Minibatch 1024</b></td><td class="rep-rec">not recorded</td><td class="rep-num">—</td><td><span class="rep-chip mid">suggestive</span></td></tr>
    <tr><td><b>EMA decay 0.9995 vs 0.999</b></td><td class="rep-rec">no successor</td><td class="rep-num">—</td><td><span class="rep-chip mid">no signal</span></td></tr>
    <tr><td><b>FP8 training</b></td><td class="rep-rec">0 – 2044 – 4</td><td class="rep-num neg">−2044</td><td><span class="rep-chip no">catastrophic</span></td></tr>
  </tbody>
</table>
</div>

Two strong results, one moderate, three that do not separate from their control, one rejection. That is a fair summary of a week of ablations, and it is a much smaller claim than "six improvements".

## The ablation that predicted itself

The last generation is the cleanest illustration of the equation above, and I did not see it until I wrote the numbers next to each other.

`adv_top_frac = 0.50` trains PPO on the top half of the advantage distribution by magnitude instead of the top quarter — twice as many selected samples from each rollout. The three measurements were: throughput fell from **26.9k to 16.6k** agent-steps per second, a 38% cut; it matched the faster control while consuming roughly **38% fewer** environment interactions; and head to head over 2,048 games it went **1009 – 1030 – 2**, a tie.

Put those in the equation. The fraction of samples still needed is \\(a = N'/N \approx 0.62\\), and the fraction of throughput retained is \\(b = R'/R = 16.6/26.9 = 0.617\\). The acceptance condition \\(a < b\\) is not satisfied — it holds *with equality*, to two significant figures. The sample-efficiency gain paid for the throughput loss exactly and bought nothing on top, so \\(T'/T \approx 1\\). The tie in the head-to-head is what the arithmetic predicted before the games were played.

I promoted 0.50 regardless, using a tie-break I had declared in advance: the larger margin against the frozen parent, +111 against +80. In a competition with a deadline that is a reasonable way to settle a coin flip. As a claim about advantage filtering it is worth nothing, and I want to be explicit that **g08 is my operational champion, not a replicated result.**

## The narrowing

The most interesting thing that happened all week is a failure I did not solve. Across three independent runs, the policy became monotonically stronger against its training distribution and monotonically worse at finishing games.

In the league run, the seven scripted opponents saturated — six of seven at 100% wins and 0% draws by iteration 725, and across the full evaluation panel the champion lineage lost *zero* games to scripted bots at any checkpoint I measured. Over the same window, greedy games against a random mover went from 1 draw in 32 at iteration 200 to 18 draws in 32 at iteration 1400. Mean owned cities rose from 3.28 to 4.87.

My first explanation was overfitting to the scripted pool, and it was wrong. The run that killed it had **no scripted opponents at all** — pure self-play against frozen copies of itself — and its draws against random still went from 1/32 to 13/32 by iteration 800, with cities rising 4.5 → 5.99 and entropy falling.

<figure class="rep-fig">
  <p class="rep-fig-title">Figure 9 &nbsp;·&nbsp; Stronger against its own distribution, worse at closing</p>
  <p class="rep-fig-sub">two measured endpoints per series — the segments are not interpolated curves</p>
  <div class="rep-scroll">
  <svg viewBox="0 0 1080 300" role="img"
       aria-label="Three panels showing that as scripted win rate saturates at 100 percent, draws against a random mover climb from 1 in 32 to 18 in 32 and mean owned cities climb from 3.28 to 4.87. A pure self-play run with no scripted opponents shows the same climb, ruling out overfitting to the scripted pool.">
    <!-- panel 1 -->
    <text class="ttl" x="0" y="12">SCRIPTED OPPONENTS</text>
    <text class="mut" x="0" y="28">win rate, 7 bots</text>
    <g transform="translate(0,44)">
      <line class="axis" x1="34" y1="160" x2="290" y2="160"/>
      <line class="axis" x1="34" y1="0" x2="34" y2="160"/>
      <text x="26" y="6"   text-anchor="end" class="mut">100%</text>
      <text x="26" y="164" text-anchor="end" class="mut">0</text>
      <line x1="60" y1="24" x2="264" y2="4" stroke="var(--blue)" stroke-width="2.5"/>
      <circle cx="60" cy="24" r="4.5" fill="var(--blue)"/>
      <circle cx="264" cy="4" r="4.5" fill="var(--blue)"/>
      <text x="264" y="-6" text-anchor="end" fill="var(--blue)" style="font-size:11px">100% · 0 losses</text>
      <text x="60" y="182" text-anchor="middle" class="mut">iter 200</text>
      <text x="264" y="182" text-anchor="middle" class="mut">1400</text>
    </g>
    <!-- panel 2 -->
    <text class="ttl" x="370" y="12">DRAWS vs A RANDOM MOVER</text>
    <text class="mut" x="370" y="28">out of 32 greedy games</text>
    <g transform="translate(370,44)">
      <line class="axis" x1="34" y1="160" x2="290" y2="160"/>
      <line class="axis" x1="34" y1="0" x2="34" y2="160"/>
      <text x="26" y="6"   text-anchor="end" class="mut">20</text>
      <text x="26" y="84"  text-anchor="end" class="mut">10</text>
      <text x="26" y="164" text-anchor="end" class="mut">0</text>
      <!-- league run 1/32 -> 18/32 -->
      <line x1="60" y1="152" x2="264" y2="16" stroke="var(--red)" stroke-width="2.5"/>
      <circle cx="60" cy="152" r="4.5" fill="var(--red)"/>
      <circle cx="264" cy="16" r="4.5" fill="var(--red)"/>
      <text x="258" y="10" text-anchor="end" fill="var(--red)" style="font-size:11px">18 / 32</text>
      <text x="72" y="148" fill="var(--red)" style="font-size:11px">1 / 32</text>
      <!-- pure self-play 1/32 -> 13/32 at iter 800 -->
      <line x1="60" y1="152" x2="176" y2="56" stroke="var(--rep-cyan)" stroke-width="2.5" stroke-dasharray="5 3"/>
      <circle cx="176" cy="56" r="4.5" fill="var(--rep-cyan)"/>
      <text x="184" y="52" fill="var(--rep-cyan)" style="font-size:11px">13 / 32</text>
      <text x="184" y="66" class="mut">no scripted pool</text>
      <text x="60" y="182" text-anchor="middle" class="mut">iter 200</text>
      <text x="264" y="182" text-anchor="middle" class="mut">1400</text>
    </g>
    <!-- panel 3 -->
    <text class="ttl" x="740" y="12">MEAN OWNED CITIES</text>
    <text class="mut" x="740" y="28">castles held at sampled timesteps</text>
    <g transform="translate(740,44)">
      <line class="axis" x1="34" y1="160" x2="290" y2="160"/>
      <line class="axis" x1="34" y1="0" x2="34" y2="160"/>
      <text x="26" y="6"   text-anchor="end" class="mut">6</text>
      <text x="26" y="84"  text-anchor="end" class="mut">3</text>
      <text x="26" y="164" text-anchor="end" class="mut">0</text>
      <line x1="60" y1="72" x2="264" y2="30" stroke="var(--red)" stroke-width="2.5"/>
      <circle cx="60" cy="72" r="4.5" fill="var(--red)"/>
      <circle cx="264" cy="30" r="4.5" fill="var(--red)"/>
      <text x="72" y="66" fill="var(--red)" style="font-size:11px">3.28</text>
      <text x="258" y="24" text-anchor="end" fill="var(--red)" style="font-size:11px">4.87</text>
      <line x1="60" y1="40" x2="176" y2="6" stroke="var(--rep-cyan)" stroke-width="2.5" stroke-dasharray="5 3"/>
      <circle cx="60" cy="40" r="4.5" fill="var(--rep-cyan)"/>
      <circle cx="176" cy="6" r="4.5" fill="var(--rep-cyan)"/>
      <text x="184" y="10" fill="var(--rep-cyan)" style="font-size:11px">4.5 → 5.99</text>
      <text x="60" y="182" text-anchor="middle" class="mut">iter 200</text>
      <text x="264" y="182" text-anchor="middle" class="mut">1400</text>
    </g>
    <text x="0" y="278" class="mut">Build rate stayed flat near 2% throughout — this is accumulation over long games, not a build-rate runaway.</text>
  </svg>
  </div>
  <figcaption><span class="fig-number">Figure 9</span>Each series is drawn between the two checkpoints I
    actually measured; the connecting segments are not interpolated learning curves. The dashed
    blue series is the run with no scripted opponents, which is what falsified the
    scripted-overfitting explanation.</figcaption>
</figure>

The third occurrence is the sharpest. In a round-robin between three arms of an identical recipe and seed that differed only in GPU count, the 2× arm at **45.9 million** agent-steps beat the 8× arm at **104.9 million**, 268 – 230 – 11. More than twice the samples, worse policy.

The mechanism I believe, and did not get to falsify: self-play rewards building — both seats build, castles convert into army, the economy compounds — but it never prices close-out *speed*, because a symmetric opponent is equally slow to finish. Build rate stays flat around 2% throughout, so this is accumulation over long games rather than a build-rate runaway. The resulting city-heavy policy cannot finish a game against a weak or erratic opponent, and "weak or erratic opponent" is exactly what the vs-random draw metric measures.

The cheapest falsification, which I ran out of time to run: anneal the build cost upward from its flat 35, or cap city count, and check whether the vs-random draw rate stops climbing while frozen-anchor win rates hold. A low-probability random-mover opponent in the training mix would also suppress the symptom, but it treats the symptom rather than the cause.

<div class="callout">
  <div class="callout-title">THE OPEN PROBLEM</div>
  <p>Self-play generates a distribution that is stationary in a way the real evaluation is not. The metric that revealed it — draw rate against a deliberately weak opponent — is one I only started logging because a champion had already fooled me.</p>
</div>

## The rejection ledger

Most of what I tried did not work, which is the normal ratio and worth writing down explicitly. Three of the rejections share a structure that I think is the most transferable methodological lesson here: **a microbenchmark measured a speed-up that the end-to-end system did not deliver.**

<figure class="rep-fig">
  <p class="rep-fig-title">Figure 10 &nbsp;·&nbsp; What the microbenchmark promised, what the run delivered</p>
  <div class="rep-scroll">
  <svg viewBox="0 0 1080 268" role="img"
       aria-label="Three optimisations on a shared axis of percentage speed-up. The HL-Gauss value lookup measured plus 49.5 percent in a synthetic microbenchmark and zero percent in the training loop. Post-step observation elision measured plus 0.94 percent on concurrent runs and minus 0.34 percent on a sequential same-GPU confirmation. FP8 training did deliver its plus 21 percent, and lost 0 to 2044 against the seed checkpoint.">
    <text class="ttl" x="0" y="12">SPEED-UP MEASURED IN ISOLATION &#8594; SPEED-UP REALISED IN TRAINING</text>
    <g class="legend-g">
      <circle cx="640" cy="8" r="5" fill="none" stroke="var(--muted)" stroke-width="1.8"/>
      <text x="652" y="12" class="mut">microbenchmark</text>
      <circle cx="792" cy="8" r="5" fill="var(--muted)"/>
      <text x="804" y="12" class="mut">end-to-end training run</text>
    </g>
    <!-- axis: -5% .. +55% maps x 320 .. 1000 -->
    <g class="grid">
      <line x1="490"   y1="40" x2="490"   y2="216"/>
      <line x1="603.3" y1="40" x2="603.3" y2="216"/>
      <line x1="716.7" y1="40" x2="716.7" y2="216"/>
      <line x1="830"   y1="40" x2="830"   y2="216"/>
      <line x1="943.3" y1="40" x2="943.3" y2="216"/>
    </g>
    <line x1="376.7" y1="34" x2="376.7" y2="216" stroke="var(--faint)" stroke-width="1.5" stroke-dasharray="4 4"/>
    <text x="376.7" y="28" text-anchor="middle" class="mut">no change</text>
    <line class="axis" x1="320" y1="216" x2="1000" y2="216"/>
    <text x="376.7" y="232" text-anchor="middle" class="mut">0</text>
    <text x="490"   y="232" text-anchor="middle" class="mut">+10</text>
    <text x="603.3" y="232" text-anchor="middle" class="mut">+20</text>
    <text x="716.7" y="232" text-anchor="middle" class="mut">+30</text>
    <text x="830"   y="232" text-anchor="middle" class="mut">+40</text>
    <text x="943.3" y="232" text-anchor="middle" class="mut">+50%</text>
    <defs>
      <marker id="ar4" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0 0 L10 5 L0 10 z" fill="currentColor"/>
      </marker>
    </defs>
    <!-- row 1 : HL-Gauss value lookup, +49.5% -> 0.0% -->
    <g color="var(--red)">
      <text x="0" y="74"  style="font-size:11.5px" fill="var(--ink)">HL-Gauss value lookup</text>
      <text x="0" y="90"  class="mut">3,050 &#8594; 4,560 samples/s, synthetic Muon update</text>
      <line x1="937.7" y1="78" x2="392" y2="78" stroke="currentColor" stroke-width="2.5" marker-end="url(#ar4)"/>
      <circle cx="937.7" cy="78" r="5.5" fill="var(--paper)" stroke="var(--red)" stroke-width="2"/>
      <circle cx="376.7" cy="78" r="5.5" fill="var(--red)"/>
      <text x="929" y="66" text-anchor="end" fill="var(--red)" style="font-size:12px">+49.5%</text>
      <text x="392" y="66" fill="var(--red)" style="font-size:12px">0.0%</text>
      <text x="440" y="66" class="mut">both arms 74.7k SPS &#183; the update was never the bottleneck</text>
    </g>
    <!-- row 2 : FP8, +21% delivered -->
    <g color="var(--red)">
      <text x="0" y="136" style="font-size:11.5px" fill="var(--ink)">FP8 training</text>
      <text x="0" y="152" class="mut">the one that kept its speed-up</text>
      <circle cx="614.7" cy="140" r="5.5" fill="var(--paper)" stroke="var(--red)" stroke-width="2"/>
      <circle cx="614.7" cy="140" r="2" fill="var(--red)"/>
      <text x="614.7" y="128" text-anchor="middle" fill="var(--red)" style="font-size:12px">+21%, delivered</text>
      <text x="640" y="144" fill="var(--red)" style="font-size:11px">&#8594; and then lost 0 &#8211; 2044 &#8211; 4 to the seed checkpoint</text>
    </g>
    <!-- row 3 : observation elision, +0.94% -> -0.34% -->
    <g color="var(--rep-cyan)">
      <text x="0" y="198" style="font-size:11.5px" fill="var(--ink)">post-step observation elision</text>
      <text x="0" y="214" class="mut">concurrent 4&#215;4070S runs, then a sequential same-GPU rerun</text>
      <line x1="387.3" y1="202" x2="368" y2="202" stroke="currentColor" stroke-width="2.5" marker-end="url(#ar4)"/>
      <circle cx="387.3" cy="202" r="5.5" fill="var(--paper)" stroke="var(--rep-cyan)" stroke-width="2"/>
      <circle cx="372.8" cy="202" r="5.5" fill="var(--rep-cyan)"/>
      <text x="400" y="190" fill="var(--rep-cyan)" style="font-size:12px">+0.94%</text>
      <text x="400" y="206" fill="var(--rep-cyan)" style="font-size:12px">&#8594; &#8722;0.34%</text>
      <text x="470" y="198" class="mut">96,526 vs 96,855 agent-steps/s &#183; below measurement noise</text>
    </g>
    <text x="0" y="262" class="mut">Every one of these was measured honestly. The microbenchmark was simply not the system.</text>
  </svg>
  </div>
  <figcaption><span class="fig-number">Figure 10</span>Three optimisations whose isolated speed-up did not
    survive contact with the training loop. The HL-Gauss lookup was a 49.5% win on a synthetic
    update and zero on matched 2×5090 runs, because the update was not the bottleneck. I now reject
    a synthetic update benchmark as a proxy for training throughput at all.</figcaption>
</figure>

The remaining rejections, briefly:

<div class="table-wrap">
<table>
  <thead><tr><th>What</th><th>Result</th><th>Read</th></tr></thead>
  <tbody>
    <tr>
      <td><b>Exact CUDA port of the CNN forward pass</b></td>
      <td>16.73 → 7.57 ms per batch-128 forward after channels-last GroupNorm, fused residuals, CUDA graphs and single-pass norm statistics — a 2.21× optimisation. JAX still did it in 6.27 ms.</td>
      <td><span class="rep-chip no">rejected</span> Six 3×3 192-channel convolutions were 68.4% of kernel time; XLA already had them. Writing my own kernels lost to the compiler by 17–20%.</td>
    </tr>
    <tr>
      <td><b>Conv+MinGRU student distilled from the champion</b></td>
      <td>16.4 M teacher-labelled steps. Greedy action agreement rose 1.6% → 10.6% against an 80% gate; policy KL stuck above 4 against a 0.20 gate.</td>
      <td><span class="rep-chip no">rejected</span> A 332k-parameter student at 10% agreement is not a compression of the teacher, it is a different policy.</td>
    </tr>
    <tr>
      <td><b>Model soup across 2×/4×/8× arms</b></td>
      <td>Soup beat the 8× arm by +44 and tied the other two.</td>
      <td><span class="rep-chip no">rejected</span> All three arms shared one recipe and one seed, so they sat in the same basin. Soups earn their gains from independent initialisations; there was no diversity to average.</td>
    </tr>
    <tr>
      <td><b>Reverse-KL "castle magnet"</b></td>
      <td>Inert — build probability did not move.</td>
      <td><span class="rep-chip no">rejected</span> Forward KL did raise build probability, and flattened the policy everywhere else. Forced-build exploration was the cleaner mechanism.</td>
    </tr>
    <tr>
      <td><b>Counterfactual paired build rollouts</b></td>
      <td>The auxiliary objective fit its replay: preference accuracy 0.33 → 0.82 over 726 iterations. Natural build rate went 0.324% → 0.264%.</td>
      <td><span class="rep-chip no">rejected</span> It learned to classify a sparse replay buffer without changing behaviour. Worse, a path bug meant the checkpoint mirror watched the wrong directory, so no weights survived the instance being destroyed.</td>
    </tr>
  </tbody>
</table>
</div>

## The bottleneck was not the algorithm

I logged every GPU I rented. Over eight days that is 67 rental records across 54 instances, and exactly one of them is marked `success` in the failure column. The other 66 carry 41 distinct failure modes: `no-ssh-in-budget` ten times, image pulls wedged for 90 minutes, broken CDI GPU mappings, a host whose advertised $2.809/hr billed at $4.79, two spot preemptions, and one instance that kept billing for two hours after two destroy calls silently aborted.

The clearest picture of what that costs is a race I ran deliberately: five machines rented in the same minute, same image, same script.

<figure class="rep-fig">
  <p class="rep-fig-title">Figure 11 &nbsp;·&nbsp; Five hosts, same minute, same image</p>
  <p class="rep-fig-sub">minutes from rental to the first training step</p>
  <div class="rep-scroll">
  <svg viewBox="0 0 1080 300" role="img"
       aria-label="A waterfall of provisioning stages for six rentals. Finland reached JAX in 1.52 minutes, North Carolina 2.04, Washington 2.27, Vietnam 4.28, and a California host never became usable and was destroyed after 13 minutes. A warm cached image on the same North Carolina host reached JAX in 0.64 minutes.">
    <!-- x: 0..6 min -> 220..980 -->
    <g class="grid">
      <line x1="220" y1="24" x2="220" y2="250"/>
      <line x1="346.7" y1="24" x2="346.7" y2="250"/>
      <line x1="473.3" y1="24" x2="473.3" y2="250"/>
      <line x1="600" y1="24" x2="600" y2="250"/>
      <line x1="726.7" y1="24" x2="726.7" y2="250"/>
      <line x1="853.3" y1="24" x2="853.3" y2="250"/>
      <line x1="980" y1="24" x2="980" y2="250"/>
    </g>
    <text x="220" y="268" text-anchor="middle" class="mut">0</text>
    <text x="346.7" y="268" text-anchor="middle" class="mut">1</text>
    <text x="473.3" y="268" text-anchor="middle" class="mut">2</text>
    <text x="600" y="268" text-anchor="middle" class="mut">3</text>
    <text x="726.7" y="268" text-anchor="middle" class="mut">4</text>
    <text x="853.3" y="268" text-anchor="middle" class="mut">5</text>
    <text x="980" y="268" text-anchor="middle" class="mut">6 min</text>
    <text class="ttl" x="0" y="16">HOST</text>
    <text class="ttl" x="220" y="16">CONTAINER → SSH → RSYNC → JAX SEES GPU → FIRST TRAINING STEP</text>
    <!-- rows: y, label, price, jaxMin, trainMin -->
    <!-- Finland -->
    <g transform="translate(0,44)">
      <text x="0" y="0" style="font-size:12px" fill="var(--ink)">Finland</text>
      <text x="0" y="15" class="mut">$1.085/hr</text>
      <rect x="220" y="-9" width="192.4" height="18" fill="var(--red)" opacity=".45" rx="1"/>
      <rect x="412.4" y="-9" width="200.1" height="18" fill="var(--blue)" opacity=".28" rx="1"/>
      <circle cx="412.4" cy="0" r="4" fill="var(--blue)"/>
      <text x="420" y="-14" fill="var(--blue)" style="font-size:10.5px">JAX 1.52</text>
      <text x="620" y="4" class="mut">training 3.10</text>
    </g>
    <!-- North Carolina -->
    <g transform="translate(0,80)">
      <text x="0" y="0" style="font-size:12px" fill="var(--ink)">N. Carolina</text>
      <text x="0" y="15" class="mut">$0.946/hr</text>
      <rect x="220" y="-9" width="258.2" height="18" fill="var(--red)" opacity=".45" rx="1"/>
      <rect x="478.2" y="-9" width="208.9" height="18" fill="var(--blue)" opacity=".28" rx="1"/>
      <circle cx="478.2" cy="0" r="4" fill="var(--blue)"/>
      <text x="486" y="-14" fill="var(--blue)" style="font-size:10.5px">2.04</text>
      <text x="694" y="4" class="mut">3.69</text>
    </g>
    <!-- Washington -->
    <g transform="translate(0,116)">
      <text x="0" y="0" style="font-size:12px" fill="var(--ink)">Washington</text>
      <text x="0" y="15" class="mut">$1.139/hr</text>
      <rect x="220" y="-9" width="287.4" height="18" fill="var(--red)" opacity=".45" rx="1"/>
      <rect x="507.4" y="-9" width="282.3" height="18" fill="var(--blue)" opacity=".28" rx="1"/>
      <circle cx="507.4" cy="0" r="4" fill="var(--blue)"/>
      <text x="515" y="-14" fill="var(--blue)" style="font-size:10.5px">2.27</text>
      <text x="797" y="4" class="mut">4.50</text>
    </g>
    <!-- Vietnam -->
    <g transform="translate(0,152)">
      <text x="0" y="0" style="font-size:12px" fill="var(--ink)">Vietnam</text>
      <text x="0" y="15" class="mut">$1.074/hr</text>
      <rect x="220" y="-9" width="541.9" height="18" fill="var(--red)" opacity=".45" rx="1"/>
      <rect x="761.9" y="-9" width="167.1" height="18" fill="var(--blue)" opacity=".28" rx="1"/>
      <circle cx="761.9" cy="0" r="4" fill="var(--blue)"/>
      <text x="769" y="-14" fill="var(--blue)" style="font-size:10.5px">4.28</text>
      <text x="936" y="4" class="mut">5.60</text>
    </g>
    <!-- California -->
    <g transform="translate(0,188)">
      <text x="0" y="0" style="font-size:12px" fill="var(--ink)">California</text>
      <text x="0" y="15" class="mut">$0.647/hr</text>
      <rect x="220" y="-9" width="760" height="18" fill="url(#hatch)" stroke="var(--red)" rx="1"/>
      <text x="600" y="4" text-anchor="middle" fill="var(--red)" style="font-size:11px">wedged 13 minutes, destroyed, never usable</text>
    </g>
    <!-- warm -->
    <g transform="translate(0,232)">
      <text x="0" y="0" style="font-size:12px" fill="var(--ink)">N. Carolina</text>
      <text x="0" y="15" class="mut">$0.959/hr · warm</text>
      <rect x="220" y="-9" width="81" height="18" fill="var(--red)" opacity=".45" rx="1"/>
      <rect x="301" y="-9" width="344.4" height="18" fill="var(--blue)" opacity=".28" rx="1"/>
      <circle cx="301" cy="0" r="4" fill="var(--blue)"/>
      <text x="309" y="-14" fill="var(--blue)" style="font-size:10.5px">0.64 — a cached image cuts time-to-GPU 3×</text>
      <text x="653" y="4" class="mut">3.36 · 77,536 SPS</text>
    </g>
  </svg>
  </div>
  <figcaption><span class="fig-number">Figure 11</span>Identical request, five providers, a 2.8× spread in
    time-to-GPU and one total loss. The 3.57 GB container image is the dominant term, which is why
    the warm-cache row is the interesting one: the same host reached JAX in 0.64 minutes instead of
    2.04.</figcaption>
</figure>

Money behaved the same way. The accountable spend across both providers is about **$63** — $49 on Vast from the durations I actually wrote down and $14 on Prime Intellect from paired create/terminate records — and that is a lower bound, because roughly half the rentals have no recorded duration. One overnight lineage died mid-run when the balance hit zero. An 8×5090 at $4.50/hr was 61% of a $7.42/hr three-box burn against a $23 credit balance, which is a sentence I could have written before renting it rather than after.

The useful measurement to come out of all this is that **the smallest box won on throughput per dollar**. In the production league workload, 2×5090 delivered 40,038 SPS per dollar-hour against 33,020 for 8×5090; in near-pure self-play, 48,916 against 37,819. Multi-GPU scaling was 3.4–3.6× across a 4× device count, and the interconnect on rented consumer hardware is where the rest went. Dropping the scripted opponent pool bought another 15–22%.

<div class="callout warning">
  <div class="callout-title">THE HONEST ACCOUNTING</div>
  <p>None of this is research. All of it is the reason the research budget was the size it was. The honest accounting of a compute-constrained week is that a large fraction of it went into making the compute exist.</p>
</div>

## What I believe, and what I do not

Three conclusions I would defend:

1. **Symmetry augmentation works when it is part of the behaviour policy.** Applied to stored samples it is a biased estimator wearing a helpful costume. +134 over the matched control in 2,048 games.
2. **Longer-horizon credit assignment materially helped.** GAE λ = 0.9 over 0.7, +172 direct and +247 against the frozen parent — the cleanest single optimisation result of the week.
3. **The 2.72 M CNN had not reached a capacity ceiling.** Every generation still improved on its parent through 1.76 billion cumulative steps, and I shipped a model that used 14% of its latency budget.

And the things this evidence cannot support, which I think matter more:

- **One seed per arm.** Everything above is a single training run per condition. Three independent 100 M-step seeds would have told me more about any one hyperparameter than one more 300 M-step continuation did.
- **Immediate-parent comparison hides non-transitivity.** I never evaluated every generation against one permanent external panel. Self-play lineages can specialise into rock-paper-scissors relationships that a parent-only ladder cannot see, and I have three arms that beat each other inconsistently as evidence that this is not hypothetical.
- **1.76 billion self-play steps is not 1.76 billion samples.** Opponents, maps, advantages and value targets all come from adjacent versions of one network. The effective sample size is far smaller and I do not know by how much.
- **A leaderboard placing is one draw from a distribution.** 9th of 114 is one submission on one ladder, and I would not read a rank difference of three places as a strength difference at all.

The version of this sweep I would run with a supervisor and a real compute allocation is not more steps. It is three seeds per arm, matched on both agent-steps and wall-clock; intermediate checkpoints rather than terminal ones; one permanent frozen anchor panel including the architectures I abandoned; and direct treatment-versus-control as the reported statistic, with the parent comparison demoted to a sanity check.

> Ablation-maxxing is an excellent way to ship a stronger agent and a poor way to explain one. In self-play RL, continued training is itself a powerful treatment, and every ablation has to prove it contributed more than another two hours of self-play would have.

The competition result was 9th of 114, on a policy that moved in 20 ms against a 150 ms budget, trained for about sixty dollars. The part I would actually defend in a seminar is none of that. It is the observation in *One equation* and its one clean instance in *The ablation that predicted itself*: on a constrained budget the two levers are steps-to-target and steps-per-second, they almost always pull against each other, and the second one is much easier to lose without noticing. That is a question I would like to answer properly — across many environments, with seeds, and with the throughput term measured rather than assumed.

<div class="rep-endnote">
  <h4>Notes on the numbers</h4>
  <p>All head-to-head records are seat-balanced and quoted as W–L–D from the first named agent's perspective. "Margin" is \(W-L\); the promotion bar is \(1.65\sqrt{W+L}\), which is ≈74 at 2,048 games. Agent-steps are counted as \(2\times\) environment ticks in a 1v1 game. Throughput figures are median <code>train/sps</code> over post-compilation iterations and are total across devices. The submission bundle for the final champion exports to numpy and matches the JAX reference at 0 action mismatches over 663 turns with a maximum logit difference of 8.2 × 10<sup>−5</sup>.</p>
  <p>Figures 9 and 10 plot only measured endpoints; the connecting segments carry no claim about the path between them. The <code>adv_top_frac</code> sample-efficiency figure of ≈38% is derived from matched-wall-clock environment interactions, not from a separate steps-to-gate measurement, and is the weakest number in the section it appears in.</p>
</div>


<script>
(function () {
  var svg = document.getElementById('w-svg');
  if (!svg) return;
  var pt = document.getElementById('w-pt'), vl = document.getElementById('w-vline'),
      hl = document.getElementById('w-hline'), hit = document.getElementById('w-hit'),
      bar = document.getElementById('w-bar'),
      big = document.getElementById('w-big'), note = document.getElementById('w-note'),
      gS = document.getElementById('w-g'), cS = document.getElementById('w-c'),
      gV = document.getElementById('w-gv'), cV = document.getElementById('w-cv');

  // plot: x 78..470 maps cost 0..80% ; y 350..30 maps gain 0..80%
  var X0 = 78, X1 = 470, Y0 = 350, Y1 = 30, MAX = 80;
  var toX = c => X0 + (c / MAX) * (X1 - X0);
  var toY = g => Y0 - (g / MAX) * (Y0 - Y1);

  function draw(gain, cost) {
    var cx = toX(Math.max(0, cost)), cy = toY(gain);
    pt.setAttribute('cx', cx); pt.setAttribute('cy', cy);
    vl.setAttribute('x1', cx); vl.setAttribute('x2', cx); vl.setAttribute('y2', cy);
    hl.setAttribute('y1', cy); hl.setAttribute('y2', cy); hl.setAttribute('x2', cx);

    // T'/T = (N'/N) / (R'/R) = (1-gain) / (1-cost)
    var a = 1 - gain / 100, b = 1 - cost / 100;
    var ratio = b <= 0 ? Infinity : a / b;
    big.textContent = !isFinite(ratio) ? '∞' : ratio.toFixed(2) + '×';

    var faster = ratio < 0.995, slower = ratio > 1.005;
    var col = faster ? 'var(--samples)' : slower ? 'var(--clock)' : 'var(--ink-2)';
    big.style.color = col; pt.setAttribute('fill', col);

    // diverging bar: centre is break-even, half-width is a 1.0 deviation either way
    var dev = Math.min(1, Math.abs((isFinite(ratio) ? ratio : 2) - 1));
    bar.style.background = col;
    bar.style.width = (dev * 50) + '%';
    if (ratio >= 1) { bar.style.left = '50%'; bar.style.right = 'auto'; }
    else { bar.style.right = '50%'; bar.style.left = 'auto'; }

    note.textContent = faster
      ? 'reaches the same strength in ' + Math.round((1 - ratio) * 100) + '% less time'
      : slower
        ? (isFinite(ratio) ? 'takes ' + Math.round((ratio - 1) * 100) + '% longer' : 'never gets there')
        : 'exactly break-even';
    gV.textContent = gain; cV.textContent = cost;
  }

  function sync() { draw(+gS.value, +cS.value); }
  gS.addEventListener('input', sync);
  cS.addEventListener('input', sync);

  document.querySelectorAll('.presets button').forEach(function (b) {
    b.addEventListener('click', function () {
      gS.value = b.dataset.g; cS.value = Math.max(0, +b.dataset.c); sync();
    });
  });

  // drag / click on the plane
  var dragging = false;
  function fromEvent(e) {
    var r = svg.getBoundingClientRect(), vb = svg.viewBox.baseVal;
    var x = (e.clientX - r.left) / r.width * vb.width;
    var y = (e.clientY - r.top) / r.height * vb.height;
    var cost = Math.round(Math.min(MAX, Math.max(0, (x - X0) / (X1 - X0) * MAX)));
    var gain = Math.round(Math.min(MAX, Math.max(0, (Y0 - y) / (Y0 - Y1) * MAX)));
    gS.value = gain; cS.value = cost; sync();
  }
  function down(e) { dragging = true; fromEvent(e); e.preventDefault(); }
  hit.addEventListener('pointerdown', down);
  pt.parentNode.addEventListener('pointerdown', down);
  window.addEventListener('pointermove', function (e) { if (dragging) fromEvent(e); });
  window.addEventListener('pointerup', function () { dragging = false; });

  sync();
})();
</script>
