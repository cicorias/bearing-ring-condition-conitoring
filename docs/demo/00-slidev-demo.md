# 00 - Slidev demo

The demo deck is in `demos/grinder-diagnostics/`. Presenter notes provide a concise talk track and
the live-demo slide lists the commands to run.

## First setup

```bash
mise run h:demo:setup
```

## Present locally

Keep the model API running in one terminal:

```bash
mise run f:api:serve
```

Start Slidev in another:

```bash
mise run i:demo:serve
```

Open the URL printed by Slidev. Press `p` for presenter mode and speaker notes.

## Build static slides

```bash
mise run j:demo:build
```

The static output is written to `demos/grinder-diagnostics/dist/` and is intentionally ignored by
Git.
