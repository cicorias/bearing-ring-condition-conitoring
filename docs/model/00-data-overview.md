# 00 - Grinder diagnostics model data

## What this dataset is

The paper studies a bearing-ring grinding machine under normal operation and five deliberately
introduced faults. Each test contains:

- 7 grinding-wheel dressing cycles;
- 15 rings per dressing cycle;
- 105 ring files in total.

The complete paper dataset has 7 tests and 735 rings.

## Source location and current coverage

The dataset inspected for this reproduction is visible from WSL at:

`/mnt/c/temp/2022-136-1-2`

It contains the research paper, a short data-description PDF, extracted data, and original ZIP
archives under `data/`.
The similar path `/mnt/c/temp/2022-136-1` does not exist.

The code does not hardcode that machine-specific path. Configure another location with
`GRINDER_DIAGNOSTICS_DATA_ROOT` in the environment. Mise loads the repository-root `.env` through
`mise.toml`; the Python settings layer also loads `.env` when invoked without mise. In direct Python
execution, an existing process environment value wins over `.env`. With the variable unset, the
fallback is `./data/source`. A command-line `--data-root` value takes highest precedence.

## Tests and labels

| Test | Machine condition | Current status |
|---|---|---|
| 1 | Normal baseline | Present: extracted and zipped |
| 2 | Workhead drive-belt damage | Present |
| 3 | Workhead spindle unbalance | Present |
| 4 | Drive-plate setup fault | Present |
| 5 | Workhead tooling setup fault | Present |
| 6 | Worn workhead tooling support | Present |
| 7 | Normal baseline | Present: extracted and zipped |

All seven tests are now present. Each extracted test and its ZIP contain 105 TDMS files across seven
dressing cycles, for the complete set of 735 rings.

| Data | Rows or files | Coverage |
|---|---:|---|
| TDMS ring signals | 735 | 105 rings in each of Tests 1 through 7 |
| `process_data.csv` | 735 | Every ring |
| `measured_quality_param.csv` | 186 | Measured subset across all tests |
| `quality_disposition.csv` | 186 | Same measured subset |

## Archive layout

Each extracted test has one extra `test_N` directory level:

```text
data/
  test_N/
    test_N/
      dresscyc_1/
        ring_1.tdms
        ...
        ring_15.tdms
      ...
      dresscyc_7/
  test_N.zip
```

The seven ZIPs total about 20.04 GB. The extracted TDMS files total about 88.82 GB. Both copies are
source data; we will not create another raw-data copy. Normal processing will read one extracted
ring file at a time.

The small tables are located at:

- `data/proc_param/proc_param/process_data.csv`
- `data/quality/quality/measured_quality_param.csv`
- `data/quality/quality/quality_disposition.csv`

Process data identifies the test, dressing cycle, and ring, then records gap position, actual
grinding time, dressing-interval counter, and rough feed. Quality files contain measured ring
properties and accepted/unaccepted flags for the measured subset.

## What one ring file contains

A small metadata read of `test_2/dresscyc_1/ring_1.tdms` found:

- 13 analogue channels sampled at 100,000 samples per second;
- 2 digital channels sampled at 10,000 samples per second;
- about 9.5 seconds of data in that example.

### Analogue channels

| Channel | Plain-English meaning |
|---|---|
| `Vib_Grind_Motor` | Grinding-motor vibration |
| `Vib_WH_Motor` | Workhead-motor vibration |
| `Vib_Grind_Spindle` | Grinding-spindle vibration |
| `Vib_WH_Z` | Workhead vibration; metadata renames this to `Vib_WH_X` |
| `Vib_WH_Y` | Workhead vibration in the other measured direction |
| `Temp_WH_Tooling` | Workhead-tooling temperature |
| `Temp_WH_Spindle` | Workhead-spindle temperature |
| `Temp_Grind_Spindle` | Grinding-spindle temperature |
| `AE_WH` | Workhead acoustic emission |
| `Force_WH` | Workhead force from the load cell |
| `Force_WH_strain` | Workhead force from the strain gauge |
| `Power_Grind_Motor` | Grinding-motor electrical power |
| `AE_Dittel_lp30` | Grinding-spindle acoustic emission |

### Digital channels

| Channel | Plain-English meaning |
|---|---|
| `AE_limit` | Acoustic-emission contact signal |
| `Trigger` | Data-acquisition synchronization trigger |

The loader must normalize the `Vib_WH_Z`/`Vib_WH_X` naming difference and verify the schema in every
test before processing.

## Known uncertainties

- The paper says signals were low-pass filtered but does not publish every cutoff frequency.
- The paper describes segmentation from acoustic emission and controller data but does not publish
  every threshold.
- The paper does not publish random seeds or enough detail to reproduce every MATLAB application
  default exactly.
- Quality measurements exist for 186 of 735 rings, so they can support analysis but cannot be
  treated as complete labels for every ring.

These are reproduction constraints. They must be recorded with the results rather than hidden by
choosing parameters that happen to improve accuracy.
