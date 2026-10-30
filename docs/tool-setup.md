# Tool Setup

Install only the backends needed for the experiment. A valid executable path does
not guarantee convergence or support for a molecule's composition.

## RDKit and utilities

`pip install -e ".[science]"` installs RDKit, ASE, NumPy and SciPy. Missing
force-field parameters are limitations, not silently repaired scientific results.

## Amesp

Obtain Amesp from its official distribution separately from this MIT repository.

```bash
export MECHCAL_AMESP_BIN="/absolute/path/to/amesp"
export MECHCAL_AMESP_TIMEOUT="300"
export MECHCAL_AMESP_NPARA="4"
export MECHCAL_AMESP_MAXCORE_MB="1000"
```

Also enable it in the run, using `--amesp` in the example. Routes include S0
recovery, vertical electronic states, orbital/transition analysis, comparisons
and bounded S0 scans. Excited-state relaxation and S1 scans remain suspended.

## Multiwfn and Amesp m2m

```bash
export MECHCAL_MULTIWFN_BIN="/absolute/path/to/Multiwfn"
export MECHCAL_AMESP_M2M_BIN="/absolute/path/to/m2m"
```

Multiwfn requires `settings.ini` next to the executable. The adapter reads matching
Amesp `.aip`, `.aop` and `.mo` files from a successful canonical gas-phase,
closed-shell singlet TDA-aTB1 calculation. Transition coefficients require
`td_tout >= 4`. Missing data returns a precondition error, not an automatic rerun.

The implemented spatial scope is at most 150 atoms and elements
C,H,N,O,F,Au,S,B,Cl,Br,Si. It returns selected-state centroids, separation,
overlap, t index and atomic/requested-fragment contributions. Indices refer to
the calculation geometry. These contributions are not ground-state charges.

## Optional QC Backends

The xTB, DFTB+, PySCF and PySOC adapters are registered when `MECHCAL_QC_CONFIG`
names a local JSON file. Planner and MCP server must inherit the same file.

```bash
cp env/qc.example.json env/qc.local.json
export MECHCAL_QC_CONFIG="$PWD/env/qc.local.json"
```

Edit the ignored local copy to enable the backends and specify executable paths.
An empty path leaves that backend unavailable. `enabled: false` disables all
external QC routes. `threads` accepts 1-8 CPUs. Requests have whole-route wall
limits; timeouts retain diagnostics.

- **xTB:** GFN2-xTB for ground-state/contact work, IPEA-xTB for vertical IP/EA,
  and aISS for initial dimer poses. Not excited-state kinetics or measured packing.
- **DFTB+:** spin-energy/SOC routes use `mio-1-1`, matching parameter revision
  and spin constants, and support neutral closed-shell CHNO. Supply parameters
  separately; do not substitute another set without validation.
- **PySCF:** install `pyscf==2.14.0` in a separate Python environment and set
  `pyscf_python`. The route fixes RHF/6-31G/CIS and compares S0/S1 gradients at
  identical coordinates. Scope: connected neutral closed-shell CHNO, 2-40 atoms.
- **PySOC:** install upstream `gaox-qd/pysoc`, commit
  `1a520c682f4851a8ec6e551a04ec5cc89ebc2894`, with its dependencies and binaries.
  Set `pysoc_python`, `pysoc_source`, DFTB and mio paths. Source hashes are checked.
  ZEFF couplings cover S0-S3 against T1-T3, not extrapolation to metal cases.

The pinned mio revision is `263e52419ce15222c864df9b3f6fbf58bfbdfa0f`.
Requirements are implemented in `mechcal/tools/dftb_spin.py` and
`mechcal/tools/pysoc_source.py`. Binaries and parameter files are not redistributed.

## MCP and interpretation

Use `--tool-backend mcp-stdio`. Macro and Microscopic workers share Planner's
capability definitions and load details on demand. Missing inputs, unavailable
backends, unsupported chemistry and execution failures remain distinct.

Neither structural heuristics nor readiness checks prove a mechanism. Compare
trends within a method; absolute Amesp/xTB/DFTB/CIS energies are not interchangeable.
