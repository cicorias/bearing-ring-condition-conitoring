# Documentation

Start here. The numbered files are ordered for a human reader.

## Trained model component

1. [Model data](model/00-data-overview.md) explains coverage, file contents, and known
   uncertainties.
2. [Reproduction plan](model/01-reproduction-plan.md) gives the path from raw TDMS files to
   PyTorch-loadable random forests.
3. [Running and results](model/02-running-and-results.md) shows measured results and CLI/API use.

## Overall system

1. [Project proposal](architecture/00-project-proposal.md) describes the complete local-to-Azure
   workflow.
2. [Model-to-agent contract](architecture/01-model-to-agent-contract.md) defines the deterministic
   handoff to the future Microsoft Agent Framework application.

## Demo

1. [Slidev demo guide](demo/00-slidev-demo.md) explains how to present and run the local demo.

## Working rules

- The source dataset stays read-only. Configure its root with
  `GRINDER_DIAGNOSTICS_DATA_ROOT`; mise loads it from `.env` when present.
  Python also loads `.env` for direct execution without mise. Otherwise the
  code uses `./data/source`.
- Code, generated features, results, and model artifacts belong in this WSL repository.
- Both extracted files and original ZIPs are present. We will not create a third raw-data copy;
  processing will read one extracted ring file at a time.
- Python and `uv` will be pinned through `mise`; project commands will run through `mise`.
- No project command will install into or run against the system Python environment.
- On the managed device, all Python packages must resolve through
  `https://packagefeedproxy.microsoft.io/pypi/simple/`, as configured in `~/.config/uv/uv.toml`.
  PyTorch is the explicit exception and uses its direct CPU-only package index.
