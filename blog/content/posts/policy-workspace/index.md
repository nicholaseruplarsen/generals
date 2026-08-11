---
title: "Inside a Small Policy"
subtitle: "Three interactive lenses for thinking about local features, global broadcasts, and action selection in a 2.72M-parameter game-playing ResNet."
description: "An interactive visual methods note for inspecting the information flow of the g08 Generals policy."
date: 2026-08-11
draft: false
author: "Nicholas Erup Larsen"
kicker: "INTERACTIVE METHODS NOTE · VISUAL PROTOTYPE"
page_script: "js/policy-workspace.js"
---

A policy is easy to score and hard to inspect. We can count wins, time inference, and remove a layer to see whether performance falls. None of that tells us what information the network makes available to a decision—or when.

This note builds three visual instruments around **g08**, the 2.72M-parameter ResNet from the main report. The architecture is real. The activation values below are deliberately **schematic**: they demonstrate the questions I would ask before presenting mechanistic claims as results.

## A map of what becomes available

The network transforms 45 observation planes through a stem, six residual blocks, and two global-context injections. The “policy lens” below treats each stage as a row and each human-readable hypothesis as a column. Select a game phase, then inspect a cell.

<figure class="workspace-figure workspace-lens" data-workspace-lens>
  <header><span>FIGURE 1 · POLICY LENS</span><strong>Which concepts become action-accessible, and when?</strong></header>
  <div class="workspace-controls" role="group" aria-label="Game phase">
    <button type="button" class="active" data-lens-phase="opening">Opening</button>
    <button type="button" data-lens-phase="contact">First contact</button>
    <button type="button" data-lens-phase="endgame">Endgame</button>
  </div>
  <div class="lens-layout">
    <div class="lens-matrix" data-lens-matrix aria-label="Layer by concept activation map"></div>
    <aside class="lens-readout" aria-live="polite"><span data-lens-coordinate>STEM × FRONTIER</span><strong data-lens-value>0.18</strong><p data-lens-copy>Local ownership boundaries are available immediately from the observation planes.</p></aside>
  </div>
  <figcaption>Schematic activations, not measured features. A real version would replace each cell with a probe score, intervention effect, or sparse-feature loading and report uncertainty.</figcaption>
</figure>

The useful question is not “does block four contain threat?” A representation can be decodable without affecting behavior. The stronger question is whether changing that representation changes the action distribution while leaving nearby information intact.

## How far can one tile speak?

A 3×3 convolution expands a tile’s receptive field by one cell in every direction. Six blocks therefore cover 13×13—not the full 21×21 board. Global injection changes the topology: mean and max pooling compress all 441 cells into one vector, and broadcast-add writes that vector back everywhere.

Choose a target tile. Increase local depth, then switch on broadcast.

<figure class="workspace-figure broadcast-lab" data-broadcast-lab>
  <header><span>FIGURE 2 · COMMUNICATION GEOMETRY</span><strong>Local propagation grows; broadcast changes the graph.</strong></header>
  <div class="broadcast-controls">
    <label>RESIDUAL BLOCKS <input type="range" min="0" max="6" value="3" data-broadcast-depth><b data-broadcast-depth-value>3</b></label>
    <button type="button" data-broadcast-toggle aria-pressed="false">Global broadcast off</button>
  </div>
  <div class="broadcast-stage">
    <div class="broadcast-board" data-broadcast-board aria-label="Interactive receptive-field board"></div>
    <div class="broadcast-readout"><span>SELECTED DECISION</span><strong data-broadcast-reach>7×7 local view</strong><p data-broadcast-copy>The far threat cannot affect this tile through the residual trunk yet.</p><div class="signal-meter"><i data-broadcast-meter></i></div></div>
  </div>
  <figcaption>Click any traversable tile to move the decision point. The red alarm marks a distant enemy stack; the blue cell marks the decision being computed.</figcaption>
</figure>

Pooling loses location. That is acceptable only because the spatial trunk keeps the “where” locally while the broadcast contributes “how much” and “is anything alarming?” This is a design bet, not a theorem: mean/max context cannot reproduce fine-grained all-pairs routing.

## Decodability is not causality

Suppose a probe says the late trunk represents “enemy threat.” The claim becomes interesting only after an intervention. Remove the candidate signal, hold the rest of the state fixed, and ask how the legal action logits move.

The workbench below is a demonstration of that experimental logic. Toggle candidate signals to construct an intervention. The bars show a schematic response, not g08 measurements.

<figure class="workspace-figure intervention-lab" data-intervention-lab>
  <header><span>FIGURE 3 · INTERVENTION WORKBENCH</span><strong>From readable feature to behaviorally relevant variable.</strong></header>
  <div class="intervention-layout">
    <div class="intervention-features">
      <span>CANDIDATE SIGNALS</span>
      <button type="button" class="active" data-feature="threat" aria-pressed="true"><i></i><b>Far threat</b><small>max-pooled alarm</small></button>
      <button type="button" class="active" data-feature="economy" aria-pressed="true"><i></i><b>Economy lead</b><small>mean-pooled balance</small></button>
      <button type="button" class="active" data-feature="certainty" aria-pressed="true"><i></i><b>General certainty</b><small>belief-state concentration</small></button>
    </div>
    <div class="intervention-actions">
      <span>ACTION PROBABILITY</span>
      <div data-action="attack"><label>Attack <b>0%</b></label><i><em></em></i></div>
      <div data-action="defend"><label>Defend <b>0%</b></label><i><em></em></i></div>
      <div data-action="expand"><label>Expand <b>0%</b></label><i><em></em></i></div>
      <div data-action="build"><label>Build <b>0%</b></label><i><em></em></i></div>
      <div data-action="pass"><label>Pass <b>0%</b></label><i><em></em></i></div>
    </div>
  </div>
  <div class="intervention-verdict"><span data-intervention-label>NATURAL FORWARD PASS</span><p data-intervention-copy>All three candidate signals are present. Remove one and inspect which actions actually move.</p></div>
  <figcaption>A measured version needs matched states, calibrated interventions, multiple seeds, and confidence intervals. A pretty bar chart does not supply causality by itself.</figcaption>
</figure>

## What these instruments would let us claim

The three views correspond to three increasingly strong statements:

1. **Availability:** a variable can be read from an intermediate state.
2. **Communication:** the architecture gives that variable a route to the decision site.
3. **Use:** intervening on the variable changes the policy in the predicted direction.

The first is a probe result. The second follows partly from architecture. The third requires an experiment. Keeping those claims separate is the difference between an interpretability visualization and an interpretability result.

<a class="return-link" href="{{< relref "/" >}}"><span>RETURN TO THE CASE STUDY</span><strong>What 1.76 Billion Steps Taught Me About Self-Play →</strong></a>
