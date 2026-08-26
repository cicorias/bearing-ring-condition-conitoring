# 01 - Grinder diagnostics model reproduction plan

## Goal

Rebuild the paper's signal-processing and classification pipeline, compare our results with its
reported numbers, and package the final two random forests for inference through PyTorch.

**The implementation scope is only the paper's two-stage random-forest production pipeline:**

- a binary model: normal versus fault;
- a five-class model: which fault from Tests 2 through 6.

The paper's SVM, decision-tree, and k-nearest-neighbour benchmark results are reference context only.
We will not recreate or ship those models.

## Step 1 - Create a reproducible project environment

- Pin Python, `uv`, and supporting tools in `mise.toml`.
- Define Python dependencies in `pyproject.toml` and lock them with `uv.lock`.
- Create `.venv` only with `uv`.
- Add short `mise` tasks for setup, data checks, feature generation, training, evaluation, and
  export. Tasks will call scripts rather than contain embedded scripts.

**Output:** another developer can clone the repository and reproduce the environment without using
system Python.

## Step 2 - Prove that the data is complete

- Build a manifest from the existing extracted files and original archives.
- Record archive size and checksum; do not create another raw-data copy.
- Confirm the observed 7 dressing cycles, 15 rings per cycle, and 105 TDMS files per test.
- Read metadata from a small sample across every test and dressing cycle.
- Confirm channel names, sample rates, data types, and usable signal lengths.
- Confirm process data has all 735 ring keys and quality data has its expected 186-ring subset.
- Stop the full reproduction run if later checks show a mismatch between extracted files and ZIP
  manifests.

**Output:** a versioned data manifest and a clear completeness report.

## Step 3 - Stream and normalize each ring

- Read one TDMS file at a time from the extracted tree; keep ZIPs as the original reference copy.
- Normalize known channel-name differences.
- Preserve `test`, `dressing_cycle`, and `ring` as identifiers.
- Validate missing values, constant signals, clipping, and unexpected lengths.
- Never write a second full copy of the raw dataset.

**Output:** validated ring records ready for signal processing.

## Step 4 - Recreate preprocessing and cycle segmentation

- Apply anti-aliasing assumptions and configurable low-pass filters by sensor type.
- Use acoustic-emission changes and the available digital markers to locate wheel contact.
- Divide each cycle into idle, force build-up, steady roughing, and spark-out.
- Discard force build-up, as the paper does.
- Save boundary diagnostics for sampled rings so segmentation can be reviewed visually.

The paper omits exact filter cutoffs and segmentation thresholds. We will keep these values in a
plain configuration file, explain how they were chosen, and run a small sensitivity check.

**Output:** three retained segments per channel: idle, steady roughing, and spark-out.

## Step 5 - Extract the paper's features

For each retained segment and each channel, calculate these ten statistics in both time and
frequency domains:

- mean;
- standard deviation;
- skewness;
- kurtosis;
- root mean square;
- peak-to-peak;
- crest factor;
- band power;
- energy;
- 90th percentile.

Remove the low-frequency trend before the FFT, matching the paper. Store the resulting feature table
in a compact columnar format inside the repository's ignored data workspace.

**Output:** one labelled feature row per ring with provenance back to its raw TDMS member.

## Step 6 - Freeze two split protocols

Run both protocols and never mix their reported results:

1. **Paper-like split:** stratified 70% training and 30% test, with five-fold cross-validation on
   training data.
2. **Leakage-resistant split:** keep each dressing cycle entirely in either training or test data.

The second result is the stronger estimate because rings from one dressing cycle share wheel
condition. Save ring IDs and random seeds before fitting feature selection or models.

**Output:** immutable split manifests for binary and fault-class experiments.

## Step 7 - Select features and sensors

- Fit feature selection on training data only.
- Recreate the paper's regularized diagonal neighbourhood component analysis as closely as the
  published detail allows.
- Rank the top 100 features and score sensors by feature rank and frequency.
- Reproduce the paper's chosen idle-segment sensor set:
  - grinding-spindle acoustic emission;
  - workhead acoustic emission;
  - grinding-motor vibration;
  - workhead-motor vibration;
  - workhead-tooling temperature.
- Confirm whether this produces the reported 58 final features.

**Output:** selected feature names, sensor ranking, and documented differences from MATLAB NCA.

## Step 8 - Train the two random forests

Train only the two-stage random-forest pipeline with the paper's published settings:

- bootstrap aggregation;
- 30 decision trees;
- at most `n - 1` decision splits per tree;
- no tuning performed merely to chase the paper's score.

Also measure the paper's two reduced sensor configurations: process-control sensors only and
condition-monitoring sensors only.

**Output:** binary and five-class random forests plus explicit MATLAB-to-Python parameter mappings.

## Step 9 - Evaluate and compare

Report:

- confusion matrices;
- accuracy;
- precision and recall by class;
- per-class and global F1;
- cross-validation results;
- paper-like and leakage-resistant results side by side.

Paper reference points:

| Experiment | Reported result |
|---|---|
| Final binary random forest | 99.54% F1 |
| Final five-class random forest | 99.68% global F1 |
| Condition-monitoring sensors only | 99.3% binary; 96.9% fault class |
| Process-control sensors only | 98.6% binary; 90.0% fault class |

Differences will be explained using data availability, split policy, preprocessing assumptions, and
MATLAB/Python implementation differences.

**Output:** machine-readable metrics and a short results document with figures.

## Step 10 - Package inference for PyTorch

A random forest is not naturally a neural-network `state_dict`. We will keep the verified Python
forest as the reference model, then encode each tree's feature index, threshold, child nodes, and
leaf probabilities as tensors in a small `torch.nn.Module`.

Package:

- preprocessing and feature-order metadata;
- binary forest tensors;
- five-class forest tensors;
- class-label mapping;
- model version and training-data manifest hash.

Save a PyTorch-loadable artifact and prove that its predicted classes and probabilities match the
reference forests on the full held-out set within a stated numerical tolerance.

**Output:** inference artifacts that load without scikit-learn and return the same predictions as
the reproduced random forests.

## Step 11 - Provide scripts and a prediction API

Ship a complete inference surface:

- a script that loads the PyTorch artifacts and predicts from a feature JSON file;
- a sample feature file with the exact required schema and feature order;
- a script that sends the sample request to the API;
- a FastAPI service with health, model-information, and prediction endpoints;
- startup checks that fail clearly when artifacts or metadata are missing.

The prediction endpoint will accept a versioned JSON feature payload and return:

- request and model identifiers;
- normal/fault prediction and probability;
- fault type and class probabilities when a fault is detected;
- a machine-readable signal indicating whether downstream analysis should run;
- warnings and provenance needed by a future generative-AI agentic pipeline.

The API remains deterministic and model-focused. A later agent can consume its structured response
for explanation, maintenance reasoning, or workflow orchestration without being coupled to training
code.

**Output:** a locally runnable API plus load, predict, and sample-client scripts.

## Completion gates

The work is complete only when:

1. all seven tests pass schema and count checks;
2. every result is tied to a saved split manifest;
3. feature selection uses training data only;
4. both split protocols are reported;
5. both random forests and both reduced-sensor configurations are evaluated;
6. the PyTorch artifacts match reference predictions;
7. CLI and API predictions agree on the supplied sample;
8. every known departure from the paper is documented.
