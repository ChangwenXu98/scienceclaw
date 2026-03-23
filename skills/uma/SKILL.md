---
name: uma
description: Run structure relaxation using Meta's UMA (Universal Materials Accelerator) via fairchem
metadata:
---

# UMA Skill

Run structure relaxation and single-point energy calculations using [UMA](https://huggingface.co/facebook/UMA) (Universal Materials Accelerator), a machine-learned interatomic potential from Meta FAIR. UMA is a fast alternative to DFT for structure relaxation, achieving near-DFT accuracy at a fraction of the cost.

Built on the [fairchem](https://github.com/facebookresearch/fairchem) framework.

## Prerequisites

- Python 3.12+
- `fairchem-core` installed (`pip install fairchem-core`)
- HuggingFace token with access to `facebook/UMA` (gated model)
  - Set `HF_TOKEN` environment variable or run `huggingface-cli login`
- GPU recommended (CUDA); CPU works but is much slower

## Scripts

### `uma_relax.py` — Relax a crystal structure
```bash
python3 {baseDir}/scripts/uma_relax.py \
  --structure path/to/structure.cif \
  --format json
```

With Materials Project ID:
```bash
python3 {baseDir}/scripts/uma_relax.py \
  --mp-id mp-149 \
  --format json
```

Full cell relaxation (cell shape + volume + positions):
```bash
python3 {baseDir}/scripts/uma_relax.py \
  --structure LaH10.cif \
  --relax-cell \
  --fmax 0.01 \
  --steps 500 \
  --format json
```

### `uma_submit.py` — Submit UMA relaxation to SLURM (GPU node)
```bash
python3 {baseDir}/scripts/uma_submit.py \
  --mp-id mp-149 \
  --partition venkvis-h100 \
  --relax-cell \
  --format json
```

### `uma_retrieve.py` — Retrieve completed job results
```bash
python3 {baseDir}/scripts/uma_retrieve.py \
  --job-id 12345 \
  --format json
```

## Parameters

| Parameter | Script | Description |
|-----------|--------|-------------|
| `--structure` | relax | Path to CIF, POSCAR, or XYZ structure file |
| `--mp-id` | relax | Materials Project ID (fetches structure automatically) |
| `--model` | relax | UMA checkpoint: `uma-s-1p1`, `uma-s-1p2`, `uma-m-1p1` (default: `uma-m-1p1`) |
| `--task` | relax | Task/DFT level: `omat`, `omol`, `omc`, `oc20`, `odac` (default: `omat`) |
| `--device` | relax | `cuda` (default) or `cpu` |
| `--relax-cell` | relax | Enable full cell relaxation via FrechetCellFilter |
| `--fmax` | relax | Force convergence threshold in eV/A (default: 0.05) |
| `--steps` | relax | Max optimizer steps (default: 200) |
| `--optimizer` | relax | ASE optimizer: `FIRE` (default) or `LBFGS` |
| `--output-traj` | relax | Path to save ASE trajectory file |
| `--output-cif` | relax | Path to save relaxed structure as CIF |
| `--format` | relax | `summary` \| `json` |
| `--dry-run` | relax | Validate inputs without running relaxation |

## Output (JSON)

### Relax
```json
{
  "status": "COMPLETED",
  "model": "uma-m-1p1",
  "task": "omat",
  "formula": "LaH10",
  "initial_energy_eV": -45.23,
  "final_energy_eV": -48.71,
  "energy_per_atom_eV": -4.43,
  "converged": true,
  "steps_taken": 87,
  "fmax_achieved": 0.032,
  "cell_relaxed": true,
  "lattice_a": 5.12,
  "lattice_b": 5.12,
  "lattice_c": 5.12,
  "volume_A3": 134.2,
  "relaxed_structure_cif": "data_LaH10\n..."
}
```

## Tasks and DFT Levels

| Task | Dataset | DFT Level | Use For |
|------|---------|-----------|---------|
| `omat` | OMat24 | PBE/PBE+U (VASP) | Inorganic materials, photovoltaics |
| `omol` | OMol25 | wB97M-V/def2-TZVPD | Organics, pharmaceuticals |
| `omc` | OMC25 | PBE+D3 (VASP) | Molecular crystals |
| `oc20` | OC20 | RPBE (VASP) | Heterogeneous catalysis |
| `odac` | ODAC23 | PBE+D3 (VASP) | Direct air capture, MOFs |

## Models

| Model | Parameters | Speed | Accuracy |
|-------|-----------|-------|----------|
| `uma-s-1p1` | 6.6M active / 150M total | Fast | Good |
| `uma-s-1p2` | 6.6M active / 290M total | Fast | Better |
| `uma-m-1p1` | 50M active / 1.4B total | Slower | Best |
