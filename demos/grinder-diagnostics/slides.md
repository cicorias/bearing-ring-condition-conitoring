---
theme: default
title: Grinder Diagnostics Model
info: Two-stage Random Forest demonstration
transition: fade-out
mdc: true
---

<div class="eyebrow">Local model component</div>

# Grinder Diagnostics Model

Two Random Forests turn raw bearing-ring grinder signals into a typed diagnostic event.

<div class="metric-grid">
  <div class="card"><div class="metric">735</div><div>measured grinding cycles</div></div>
  <div class="card"><div class="metric">2</div><div>production Random Forests</div></div>
  <div class="card"><div class="metric">0.0</div><div>PyTorch parity delta</div></div>
</div>

<!--
Say: This demo is one deterministic component, not the whole maintenance system.
It reads measured cycle features and emits a compact result that an agent workflow can consume.
-->

---

<div class="eyebrow">Scope boundary</div>

# One component in a larger workflow

<div class="flow-grid">
  <div class="flow-step">Raw TDMS<br/>signals</div>
  <div class="flow-step">Feature<br/>pipeline</div>
  <div class="flow-step accent">Diagnostics<br/>model API</div>
  <div class="flow-step">Future agent<br/>workflow</div>
</div>

<div class="card accent" style="margin-top: 2rem">
Raw signals stop at the deterministic model boundary. Agents receive typed predictions, probabilities,
provenance, and a downstream-analysis flag.
</div>

<!--
Say: The future Microsoft Agent Framework app does not reinterpret waveforms.
It consumes the model's typed output and adds bounded maintenance reasoning and approval.
-->

---

<div class="eyebrow">Source data</div>

# Complete experimental dataset

<div class="metric-grid">
  <div class="card"><div class="metric">7</div><div>test conditions</div></div>
  <div class="card"><div class="metric">7</div><div>dressing cycles per test</div></div>
  <div class="card"><div class="metric">15</div><div>rings per dressing cycle</div></div>
</div>

- Tests 1 and 7: healthy baseline
- Tests 2–6: five induced machine faults
- 13 analogue channels at 100 kHz
- Process parameters for every ring

<!--
Say: Every ring is represented once. We validated extracted files against the seven original ZIPs
before training.
-->

---

<div class="eyebrow">Inference path</div>

# Two-stage Random Forest

<div class="contract-grid">
  <div class="card accent">
    <h2>1. Detection</h2>
    <p>Normal or fault?</p>
    <p class="muted">Healthy cycles stop here.</p>
  </div>
  <div class="card">
    <h2>2. Identification</h2>
    <p>Which of five known faults?</p>
    <p class="muted">Runs only after fault detection.</p>
  </div>
</div>

The production artifact contains 30 trees per forest and 58 selected idle-segment features.

<!--
Say: The idle segment occurs before grinding-wheel contact, so the result can arrive before another
ring is cut.
-->

---

<div class="eyebrow">Measured results</div>

# Reproduction results

| Split | Binary accuracy | Five-class accuracy |
|---|---:|---:|
| Paper-like stratified 70/30 | **98.64%** | **100.00%** |
| Dressing-cycle grouped | **99.05%** | **98.00%** |

<div class="card" style="margin-top: 1.5rem">
The grouped split is the stronger generalization check because rings sharing wheel condition remain
together.
</div>

<!--
Say: We report both. The paper-like split is comparable to the publication; the grouped split is
more conservative about leakage.
-->

---

<div class="eyebrow">Live demo</div>

# Run the local model

<div class="command">

```bash
mise run e:model:predict
```

</div>

Then start the service:

<div class="command">

```bash
mise run f:api:serve
```

</div>

And send the generated sample:

<div class="command">

```bash
mise run g:api:sample
```

</div>

<!--
Demo: Run these commands in separate terminals. Point out that CLI and HTTP responses are identical
because both use the same PyTorch artifact and inference engine.
-->

---

<div class="eyebrow">API contract</div>

# Output prepared for downstream agents

```json
{
  "model_version": "rf-…",
  "has_fault": true,
  "binary_probability": 0.97,
  "fault_type": "drive_plate_setup_fault",
  "fault_probabilities": {
    "drive_plate_setup_fault": 0.91,
    "workhead_spindle_unbalance": 0.06
  },
  "downstream_analysis_required": true,
  "provenance": {}
}
```

<div class="card accent">
The future agent branches on <code>downstream_analysis_required</code> and receives evidence—not raw
sensor arrays.
</div>

<!--
Say: This is the stable seam. Agent prompts, orchestration, and Azure deployment can change without
changing model training.
-->

---

<div class="eyebrow">Repository roadmap</div>

# Local now, Azure later

| Segment | Location |
|---|---|
| Trained model | `src/grinder_diagnostics_model/` |
| Model operations | `scripts/model/` |
| Demo | `demos/grinder-diagnostics/` |
| Future MAF application | `apps/maintenance-agent/` |
| Future azd infrastructure | `infra/azure/` |

<div class="card" style="margin-top: 1.5rem">
The model is already independently runnable. Agent and cloud layers can be added without folding
their concerns into the model package.
</div>

<!--
Close: The next architectural increment is the Microsoft Agent Framework application consuming this
API locally. Azure deployment follows only after that contract is stable.
-->
