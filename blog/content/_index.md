---
title: "What 1.76 Billion Steps Taught Me About Self-Play"
subtitle: "A small ResNet, a moving opponent, and the experiments that survived a compute-constrained RL tournament."
description: "An interactive technical report on training a 2.7M-parameter global-context ResNet for the 2026 Generals competition."
date: 2026-08-09
draft: false
author: "Nicholas Erup Larsen"
kicker: "GENERALS · AN INTERACTIVE RL CASE STUDY"
---

Reinforcement learning is easy to describe and expensive to disbelieve. Give an agent a reward, let it play itself, and wait for strategy to emerge. The awkward part is that self-play also lets the agent write its own exam.

I spent the summer finding out how much trouble that causes. I built a JAX training stack for **Generals**, an imperfect-information strategy game in which two players move simultaneously and may wait hundreds of decisions for a clean reward. My final 2.72M-parameter ResNet reached **#10 of 115**.

What follows is the useful part: the mechanisms I tried, the measurements that changed my mind, and the conclusions the evidence can actually support.

<div class="result-strip">
  <div><span>LIVE RANK</span><strong>#10<small>/ 115</small></strong></div>
  <div><span>ELO</span><strong>2315</strong></div>
  <div><span>TRAINING</span><strong>1.764B<small> steps</small></strong></div>
  <div><span>POLICY</span><strong>2.72M<small> params</small></strong></div>
</div>

{{< generals-game mode="human" >}}

This is the submitted **g08 policy**, exported from Equinox to ONNX. **Play g08** gives you the blue pieces; **g08 self-play** gives each copy its own legal fog memory. The browser export preserves the JAX argmax with maximum random-input logit error \(2.93 \times 10^{-5}\).

## The game is the research problem

The rules fit in a paragraph. The learning problem did not.

The board is random and **18–21 cells per axis**. A move sends half or all-but-one army to a neighboring tile. Territory produces army; generals and built castles produce faster. Capture the enemy general and you win. Everything difficult follows from four wrinkles:

<div class="rules-grid rule-lab" data-rule-lab>
  <button class="active" data-rule="fog"><b>01</b><span>FOG</span><small>memory is part of state</small></button>
  <button data-rule="collision"><b>02</b><span>COLLISIONS</span><small>moves resolve together</small></button>
  <button data-rule="economy"><b>03</b><span>BUILD</span><small>35+ army now, income later</small></button>
  <button data-rule="deadline"><b>04</b><span>DEATHTOUCH</span><small>waiting expires at turn 800</small></button>
  <div class="rule-stage">
    <div class="rule-board" data-rule-board aria-hidden="true"></div>
    <div><strong data-rule-title>Observation is history-dependent</strong><p data-rule-copy>Enemy stacks vanish outside a one-cell visibility halo. Two identical visible frames can require different actions because their hidden histories differ.</p></div>
  </div>
</div>

So one terminal reward must answer three questions. Which of **4,410 legalizable actions** mattered? Was spending 35 army on a castle worth being weaker now? Did the policy improve, or did both copies merely agree on a new self-play convention?

Deployment supplied a fourth constraint: one CPU, 2 GB RAM, and 150 ms per move. g08 takes about **20.7 ms** on the evaluation machine.

## Forty-five planes go in

The network never receives the hidden board. It receives **45 planes on a padded 21×21 grid**. Select a group to see what each part contributes.

<div class="channel-atlas" data-channel-atlas>
  <button class="channel-group present active" data-channel="present"><span>NOW · 0–20</span><b>armies · ownership · fog · terrain · global totals · turn phase</b></button>
  <button class="channel-group memory" data-channel="memory"><span>MEMORY · 21–30</span><b>last enemy army · time unseen · general candidates · deathtouch clocks</b></button>
  <button class="channel-group motion" data-channel="motion"><span>MOTION · 31–44</span><b>seven own-army deltas · seven enemy-army deltas</b></button>
  <div class="channel-detail"><b data-channel-count>21 planes</b><span data-channel-copy>The current legal observation: normalized armies, ownership masks, known structures, visibility, scores, board shape, and phase clocks.</span></div>
</div>

One plane deserves explanation. The enemy general must spawn at least 17 traversable steps away, so I begin with every legal candidate and eliminate cells as the game reveals them. This is a hand-built belief update, not a learned world model. It spends rules knowledge to save representation capacity.

## Six local blocks, two global messages

A convolution is naturally good at the question “what is near me?” Generals also asks “am I winning everywhere else?” Six 3×3 residual blocks see only a 13×13 neighborhood, so g08 broadcasts a pooled whole-board message after blocks two and five. Click a stage to trace the actual network.

<div class="architecture-lab">
  <div class="arch-flow" role="group" aria-label="Interactive network architecture">
    <button type="button" data-arch="input" class="active"><span>01</span><b>45-channel state</b><small>45×21×21</small></button><i>→</i>
    <button type="button" data-arch="stem"><span>02</span><b>Stem + tape</b><small>192×21×21</small></button><i>→</i>
    <button type="button" data-arch="local"><span>03</span><b>Residual trunk</b><small>6 × 3×3</small></button><i>↗</i>
    <button type="button" data-arch="global"><span>04</span><b>Global inject</b><small>2 × broadcast</small></button><i>→</i>
    <div class="arch-heads"><button type="button" data-arch="policy"><span>05A</span><b>Policy</b><small>4,410 logits</small></button><button type="button" data-arch="value"><span>05B</span><b>Value</b><small>128 bins</small></button></div>
  </div>
  <div class="arch-copy" data-arch-copy><strong>45 × 21 × 21</strong><span>Present state, motion history, fog memory, general candidates, and phase clocks.</span></div>
  <div class="receptive-field" data-receptive><div class="rf-board" aria-hidden="true"></div><label>LOCAL BLOCKS <input type="range" min="0" max="6" value="6" data-rf-range><b data-rf-label>13×13</b></label><p>A 3×3 block grows the local receptive field by two. Global injection broadcasts mean/max pooled evidence across all 441 cells.</p></div>
</div>

Each block uses GroupNorm → SiLU → 3×3 convolution with an identity skip. At blocks two and five, every tile receives the same summary vector:

<div class="formula">\(x \leftarrow x + \operatorname{MLP}([\operatorname{mean}_{cells}(x)\,\Vert\,\operatorname{max}_{cells}(x)])\)<small>mean asks “how much?”; max asks “is anything unusually salient?”; the spatial map retains “where?”</small></div>

The policy head returns ten planes per tile: four full moves, four half moves, pass, and build. A rules-derived mask deletes impossible actions before sampling. The critic predicts a 128-bin return distribution instead of one scalar.

## A win has to travel backward

Suppose a move matters, but the general falls twelve turns later. Generalized Advantage Estimation decides how much of that later surprise reaches the earlier logit. Move the delay and \(\lambda\):

<div class="credit-lab" data-credit-lab>
  <div class="credit-controls"><label>DELAY <input data-delay type="range" min="1" max="24" value="12"><b data-delay-value>12 turns</b></label><label>GAE λ <input data-lambda type="range" min="50" max="100" value="90"><b data-lambda-value>0.90</b></label></div>
  <div class="credit-bars" data-credit-bars aria-label="GAE credit weights"></div>
  <div class="credit-equation">A<sub>t</sub> = Σ (γλ)<sup>l</sup> δ<sub>t+l</sub><span data-credit-readout>12-turn signal retains 28.2% weight</span></div>
</div>

The critic supplies a baseline; GAE carries temporal-difference surprise backward; PPO adjusts the action probability while clipping the likelihood ratio. The final recipe used \(\gamma=1\), \(\lambda=0.9\), two passes per rollout, the top 50% by \(|A|\), Muon at \(10^{-4}\), and EMA weights for evaluation.

## Self-play will lie to you

The learner and its opponent are the same moving object. A smooth self-play curve can therefore mean improvement, collapse, or a mutually compatible convention. I stopped treating it as a scoreboard.

Every candidate instead faced frozen anchors in **2,048 seat-balanced games**. Promotion required \(W-L \geq 1.65\sqrt{W+L}\). Select an experiment below; the dark bar is the result that changed the next run.

<div class="experiment-console" data-experiment-console>
  <div class="experiment-tabs" role="tablist">
    <button class="active" data-experiment="d4">D4 FIX</button><button data-experiment="reuse">SAMPLE REUSE</button><button data-experiment="expert">EXPERT OPPONENT</button><button data-experiment="unet">LARGE U-NET</button><button data-experiment="drift">SELF-PLAY DRIFT</button><button data-experiment="build">BUILD REPLAY</button>
  </div>
  <div class="experiment-view">
    <div class="experiment-visual" data-experiment-visual aria-hidden="true"></div>
    <div class="experiment-copy"><span data-experiment-status>WORKED · G03</span><h3 data-experiment-title>Put symmetry in the behavior policy</h3><p data-experiment-body>The first version rotated samples only inside PPO updates. Because the CNN is not equivariant, the stored action probability no longer described the policy that generated the trajectory. The correction sampled one D4 frame per episode and acted inside it.</p><b data-experiment-inference>Inference: symmetry helped once PPO's on-policy likelihood ratio was valid.</b></div>
  </div>
</div>

## What survived the ablations

Three claims survived contact with fixed opponents:

1. **Self-play is an endogenous benchmark.** Its opponent and state distribution move with the learner. Fixed anchors converted smooth curves into claims that could fail.
2. **Exact invariances still require correct probability bookkeeping.** D4 helped only when the behavior policy acted in the transformed frame and stored that exact action/log-probability.
3. **A learned auxiliary target need not alter behavior.** Counterfactual build-preference accuracy reached ~0.82 while natural build rate fell. Most paired interventions shared the same terminal outcome, so the buffer taught a predictable label with little causal leverage.

The architectural claim is deliberately narrower: a **2.7M global-context CNN was sufficient for top ten under the CPU limit**. It never learned to beat the frozen transformer champion. That result confounds architecture with exploration, credit assignment, and curriculum, so “attention would have fixed it” remains a hypothesis.

## The next three experiments

<div class="next-grid">
  <article><span>01</span><h3>Privileged belief training</h3><p>Predict hidden occupancy from simulator state during training; discard the head at inference. Test fixed-anchor strength, not auxiliary accuracy.</p></article>
  <article><span>02</span><h3>Context interventions</h3><p>Ablate each pooled broadcast on real states and measure action/value changes by phase. Directly test what global context contributes.</p></article>
  <article><span>03</span><h3>Exploitability curriculum</h3><p>Sample frozen best responses that expose weaknesses while keeping promotion anchors fixed.</p></article>
</div>

<div class="closing-card"><span>FINAL SUBMISSION · G08</span><strong>1.764 billion agent-steps</strong><p>Global-context ResNet · fog memory · rollout-D4 · Muon · pure self-play · 20.7 ms CPU inference</p></div>

<a class="companion-link" href="{{< relref "posts/policy-workspace" >}}"><span>INTERACTIVE METHODS NOTE</span><strong>Inside a Small Policy</strong><span class="companion-copy">Explore a layer-by-concept lens, the geometry of global broadcast, and a causal intervention workbench.</span><b>Open companion article →</b></a>
