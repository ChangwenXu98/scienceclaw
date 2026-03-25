#!/usr/bin/env python3
"""
uma_screen.py — Hydride superconductor screening pipeline.

Enumerates MHx structures from prototypes, relaxes each with UMA at multiple
pressures, computes formation energies, and checks 0 GPa stability against
the Materials Project convex hull.

ScienceClaw skill contract: argparse, --format json, JSON on stdout.
Progress/diagnostics go to stderr.
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GPA_TO_EV_PER_A3 = 1.0 / 160.21766208

# Elemental reference lattice parameters
ELEMENT_REFS = {
    "La": {"crystal": "fcc", "a": 5.31},
    "Y":  {"crystal": "hcp", "a": 3.65, "c": 5.73},
    "Ca": {"crystal": "fcc", "a": 5.58},
    "Ce": {"crystal": "fcc", "a": 5.16},
    "Sc": {"crystal": "hcp", "a": 3.31, "c": 5.27},
}

# Prototypes defined by spacegroup + Wyckoff positions
PROTOTYPES = {
    6: {
        "label": "CaH6-type",
        "spacegroup": 229,   # Im-3m
        "a": 3.54,
        "species": ["M", "H"],
        "coords": [
            [0.0, 0.0, 0.0],       # 2a  — metal
            [0.25, 0.0, 0.5],       # 12d — hydrogen
        ],
    },
    10: {
        "label": "LaH10-type",
        "spacegroup": 225,   # Fm-3m
        "a": 5.10,
        "species": ["M", "H", "H"],
        "coords": [
            [0.0, 0.0, 0.0],           # 4a  — metal
            [0.25, 0.25, 0.25],         # 8c  — hydrogen
            [0.118, 0.118, 0.118],      # 32f — hydrogen (clathrate cage)
        ],
    },
    # YH9 P63/mmc (hexagonal) intentionally skipped — uniformly unstable
}


def log(msg: str) -> None:
    """Print progress to stderr."""
    print(f"[uma_screen] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Structure builders
# ---------------------------------------------------------------------------

def build_prototype(metal: str, stoich: int) -> "pymatgen.core.Structure":
    """Build an MHx structure from a prototype using pymatgen."""
    from pymatgen.core import Structure, Lattice

    proto = PROTOTYPES[stoich]
    species = [metal if s == "M" else s for s in proto["species"]]
    lattice = Lattice.cubic(proto["a"])
    structure = Structure.from_spacegroup(
        proto["spacegroup"],
        lattice,
        species,
        proto["coords"],
    )
    return structure


def pymatgen_to_ase(structure) -> "ase.Atoms":
    """Convert a pymatgen Structure to an ASE Atoms object."""
    from ase import Atoms

    atoms = Atoms(
        symbols=[str(s) for s in structure.species],
        positions=structure.cart_coords,
        cell=structure.lattice.matrix,
        pbc=True,
    )
    return atoms


def build_element_reference(symbol: str) -> "ase.Atoms":
    """Build elemental bulk reference with known lattice constants."""
    from ase.build import bulk

    ref = ELEMENT_REFS[symbol]
    if ref["crystal"] == "hcp":
        atoms = bulk(symbol, ref["crystal"], a=ref["a"], c=ref["c"])
    else:
        atoms = bulk(symbol, ref["crystal"], a=ref["a"])
    return atoms


def build_h2_reference() -> "ase.Atoms":
    """Build an H2 molecule in a 10 A box."""
    from ase import Atoms

    h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], pbc=True)
    h2.set_cell([10.0, 10.0, 10.0])
    h2.center()
    return h2


# ---------------------------------------------------------------------------
# Relaxation
# ---------------------------------------------------------------------------

def relax_atoms(atoms, predictor, pressure_gpa: float,
                fmax: float, steps: int) -> dict:
    """
    Relax an ASE Atoms object using UMA via FrechetCellFilter.

    Returns a dict with energy, convergence info, etc.
    """
    from ase.optimize import FIRE
    from ase.filters import FrechetCellFilter
    from fairchem.core import FAIRChemCalculator

    calc = FAIRChemCalculator(predictor, task_name="omat")
    atoms.calc = calc

    pressure_ev_a3 = pressure_gpa * GPA_TO_EV_PER_A3
    filtered = FrechetCellFilter(atoms, scalar_pressure=pressure_ev_a3)
    opt = FIRE(filtered, logfile=None)

    t0 = time.time()
    converged = opt.run(fmax=fmax, steps=steps)
    elapsed = time.time() - t0

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    import numpy as np
    fmax_achieved = float(np.max(np.linalg.norm(forces, axis=1)))

    return {
        "energy_eV": float(energy),
        "energy_per_atom_eV": float(energy / len(atoms)),
        "converged": bool(converged),
        "steps_taken": opt.nsteps,
        "fmax_achieved": round(fmax_achieved, 6),
        "volume_A3": float(atoms.get_volume()),
        "n_atoms": len(atoms),
        "elapsed_s": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Formation energy
# ---------------------------------------------------------------------------

def compute_formation_energy(
    e_hydride_per_atom: float,
    n_metal: int,
    n_h: int,
    e_metal_per_atom: float,
    e_h2_total: float,
) -> float:
    """
    Formation energy per atom (eV/atom).

    E_f = [E(MHx) - n_M * e_M - (n_H / 2) * E(H2)] / (n_M + n_H)
    """
    n_total = n_metal + n_h
    e_total = e_hydride_per_atom * n_total
    e_f = (e_total - n_metal * e_metal_per_atom - (n_h / 2.0) * e_h2_total) / n_total
    return e_f


# ---------------------------------------------------------------------------
# Convex hull
# ---------------------------------------------------------------------------

def get_e_above_hull(metal: str, formula: str, total_energy_eV: float) -> float | None:
    """
    Query MP for the M-H chemical system and compute energy above hull.

    Returns eV/atom or None if MP query fails.
    """
    try:
        from mp_api.client import MPRester
        from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
        from pymatgen.core import Composition

        api_key = os.environ.get("MP_API_KEY")
        if not api_key:
            log("WARNING: MP_API_KEY not set, skipping convex hull check")
            return None

        chemsys = "-".join(sorted([metal, "H"]))
        log(f"  Fetching MP entries for {chemsys} ...")

        with MPRester(api_key) as mpr:
            entries = mpr.get_entries_in_chemsys(chemsys)

        my_entry = PDEntry(Composition(formula), total_energy_eV, name=f"UMA-{formula}")
        all_entries = list(entries) + [my_entry]
        pd = PhaseDiagram(all_entries)
        e_hull = pd.get_e_above_hull(my_entry)
        return float(e_hull)
    except Exception as exc:
        log(f"  Convex hull failed for {formula}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args) -> dict:
    """Execute the full screening pipeline."""
    from fairchem.core import pretrained_mlip

    metals = [m.strip() for m in args.metals.split(",")]
    stoichs = [int(s.strip()) for s in args.stoichiometries.split(",")]
    pressures = [float(p.strip()) for p in args.pressures.split(",")]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Validate metals
    # ------------------------------------------------------------------
    for m in metals:
        if m not in ELEMENT_REFS:
            raise ValueError(
                f"Unknown metal '{m}'. Supported: {list(ELEMENT_REFS.keys())}"
            )

    # Validate stoichiometries
    for s in stoichs:
        if s not in PROTOTYPES:
            raise ValueError(
                f"No prototype for MH{s}. Available: {list(PROTOTYPES.keys())}"
            )

    # ------------------------------------------------------------------
    # Load UMA predictor ONCE
    # ------------------------------------------------------------------
    log(f"Loading UMA model '{args.model}' on {args.device} ...")
    predictor = pretrained_mlip.get_predict_unit(args.model, device=args.device)
    log("Model loaded.")

    # ------------------------------------------------------------------
    # Compute elemental reference energies (at 0 GPa)
    # ------------------------------------------------------------------
    log("Computing elemental reference energies ...")
    metal_ref_energies: dict[str, float] = {}
    for m in metals:
        log(f"  Relaxing {m} bulk reference ...")
        ref_atoms = build_element_reference(m)
        ref_result = relax_atoms(ref_atoms, predictor, 0.0, args.fmax, args.steps)
        metal_ref_energies[m] = ref_result["energy_per_atom_eV"]
        log(f"    {m}: {ref_result['energy_per_atom_eV']:.4f} eV/atom")

    log("  Relaxing H2 reference ...")
    h2_atoms = build_h2_reference()
    h2_result = relax_atoms(h2_atoms, predictor, 0.0, args.fmax, args.steps)
    e_h2_total = h2_result["energy_eV"]
    log(f"    H2: {e_h2_total:.4f} eV (molecule)")

    # ------------------------------------------------------------------
    # Enumerate and relax hydrides
    # ------------------------------------------------------------------
    results = []

    for metal in metals:
        for stoich in stoichs:
            formula = f"{metal}H{stoich}"
            proto = PROTOTYPES[stoich]
            log(f"Building {formula} ({proto['label']}) ...")

            try:
                structure = build_prototype(metal, stoich)
            except Exception as exc:
                log(f"  ERROR building {formula}: {exc}")
                results.append({
                    "formula": formula,
                    "prototype": proto["label"],
                    "status": "BUILD_FAILED",
                    "error": str(exc),
                })
                continue

            atoms_template = pymatgen_to_ase(structure)
            n_metal = sum(1 for s in atoms_template.get_chemical_symbols() if s != "H")
            n_h = sum(1 for s in atoms_template.get_chemical_symbols() if s == "H")

            pressure_data = {}

            for pressure in pressures:
                label = f"{formula}_P{pressure:.0f}GPa"
                log(f"  Relaxing {label} ...")

                atoms = atoms_template.copy()
                try:
                    relax_result = relax_atoms(
                        atoms, predictor, pressure, args.fmax, args.steps
                    )
                except Exception as exc:
                    log(f"    ERROR: {exc}")
                    pressure_data[pressure] = {
                        "status": "RELAX_FAILED",
                        "error": str(exc),
                    }
                    continue

                # Save relaxed CIF
                cif_path = output_dir / f"{label}.cif"
                try:
                    from ase.io import write as ase_write
                    ase_write(str(cif_path), atoms, format="cif")
                    relax_result["cif_path"] = str(cif_path)
                except Exception as exc:
                    log(f"    WARNING: could not save CIF: {exc}")

                # Formation energy
                e_form = compute_formation_energy(
                    relax_result["energy_per_atom_eV"],
                    n_metal, n_h,
                    metal_ref_energies[metal],
                    e_h2_total,
                )
                relax_result["formation_energy_eV_per_atom"] = round(e_form, 6)
                relax_result["pressure_GPa"] = pressure
                relax_result["status"] = "COMPLETED"

                # Convex hull at 0 GPa only
                if pressure == 0.0:
                    e_hull = get_e_above_hull(
                        metal, formula,
                        relax_result["energy_eV"],
                    )
                    relax_result["e_above_hull_eV_per_atom"] = (
                        round(e_hull, 6) if e_hull is not None else None
                    )

                pressure_data[pressure] = relax_result
                log(
                    f"    E={relax_result['energy_per_atom_eV']:.4f} eV/atom  "
                    f"Ef={e_form:.4f} eV/atom  "
                    f"{'converged' if relax_result['converged'] else 'NOT converged'}  "
                    f"({relax_result['elapsed_s']:.1f}s)"
                )

            results.append({
                "formula": formula,
                "prototype": proto["label"],
                "n_metal": n_metal,
                "n_h": n_h,
                "n_atoms": n_metal + n_h,
                "pressures": {
                    f"{p:.0f}": pressure_data[p]
                    for p in sorted(pressure_data.keys())
                },
            })

    # ------------------------------------------------------------------
    # Rank by formation energy at highest pressure
    # ------------------------------------------------------------------
    max_pressure = max(pressures)
    p_key = f"{max_pressure:.0f}"

    def sort_key(r):
        pd = r.get("pressures", {}).get(p_key, {})
        ef = pd.get("formation_energy_eV_per_atom")
        if ef is None:
            return 999.0
        return ef

    results.sort(key=sort_key)

    # Build ranking summary
    ranking = []
    for rank, r in enumerate(results, 1):
        pd = r.get("pressures", {}).get(p_key, {})
        entry = {
            "rank": rank,
            "formula": r["formula"],
            "prototype": r["prototype"],
        }
        ef = pd.get("formation_energy_eV_per_atom")
        if ef is not None:
            entry["formation_energy_eV_per_atom"] = ef
            entry["converged"] = pd.get("converged", False)
        # Include e_above_hull if available from 0 GPa data
        hull_data = r.get("pressures", {}).get("0", {})
        if hull_data.get("e_above_hull_eV_per_atom") is not None:
            entry["e_above_hull_eV_per_atom_0GPa"] = hull_data["e_above_hull_eV_per_atom"]
        ranking.append(entry)

    return {
        "status": "COMPLETED",
        "model": args.model,
        "device": args.device,
        "metals": metals,
        "stoichiometries": stoichs,
        "pressures_GPa": pressures,
        "fmax": args.fmax,
        "max_steps": args.steps,
        "output_dir": str(output_dir),
        "ranking": ranking,
        "candidates": results,
        "reference_energies": {
            "metals_eV_per_atom": metal_ref_energies,
            "H2_eV": e_h2_total,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Hydride superconductor screening with UMA",
    )
    parser.add_argument(
        "--metals", type=str, default="La,Y,Ca,Ce,Sc",
        help="Comma-separated list of metals to screen (default: La,Y,Ca,Ce,Sc)",
    )
    parser.add_argument(
        "--stoichiometries", type=str, default="6,10",
        help="Comma-separated hydrogen stoichiometries, e.g. 6,10 (default: 6,10)",
    )
    parser.add_argument(
        "--pressures", type=str, default="0,150",
        help="Comma-separated pressures in GPa (default: 0,150)",
    )
    parser.add_argument(
        "--model", type=str, default="uma-m-1p1",
        choices=["uma-s-1p1", "uma-s-1p2", "uma-m-1p1"],
        help="UMA checkpoint (default: uma-m-1p1)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device: cuda or cpu (default: cuda)",
    )
    parser.add_argument(
        "--fmax", type=float, default=0.05,
        help="Force convergence threshold in eV/A (default: 0.05)",
    )
    parser.add_argument(
        "--steps", type=int, default=200,
        help="Max optimizer steps per relaxation (default: 200)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./uma_screen_output",
        help="Directory for relaxed CIF files (default: ./uma_screen_output)",
    )
    parser.add_argument(
        "--format", type=str, default="json", choices=["json", "summary"],
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate inputs and show plan without running relaxations",
    )

    args = parser.parse_args()

    # Auto-detect GPU availability; if no GPU, submit to SLURM
    needs_slurm = False
    if args.device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                needs_slurm = True
        except ImportError:
            needs_slurm = True

    if needs_slurm and not args.dry_run:
        _submit_to_slurm(args)
        return

    metals = [m.strip() for m in args.metals.split(",")]
    stoichs = [int(s.strip()) for s in args.stoichiometries.split(",")]
    pressures = [float(p.strip()) for p in args.pressures.split(",")]

    # ------------------------------------------------------------------
    # Dry run — show plan and exit
    # ------------------------------------------------------------------
    if args.dry_run:
        candidates = []
        for metal in metals:
            for stoich in stoichs:
                if stoich in PROTOTYPES:
                    candidates.append({
                        "formula": f"{metal}H{stoich}",
                        "prototype": PROTOTYPES[stoich]["label"],
                    })

        plan = {
            "status": "DRY_RUN",
            "model": args.model,
            "device": args.device,
            "metals": metals,
            "stoichiometries": stoichs,
            "pressures_GPa": pressures,
            "fmax": args.fmax,
            "max_steps": args.steps,
            "output_dir": args.output_dir,
            "n_candidates": len(candidates),
            "n_relaxations": len(candidates) * len(pressures),
            "n_reference_relaxations": len(metals) + 1,  # metals + H2
            "candidates": candidates,
            "note": "YH9 P63/mmc skipped — uniformly unstable in prior screening",
        }
        print(json.dumps(plan, indent=2))
        return

    # ------------------------------------------------------------------
    # Run pipeline
    # ------------------------------------------------------------------
    try:
        output = run_pipeline(args)
    except Exception as exc:
        error_output = {
            "status": "FAILED",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(json.dumps(error_output, indent=2))
        sys.exit(1)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    if args.format == "json":
        print(json.dumps(output, indent=2))
    else:
        # Summary format
        print(f"UMA Hydride Screening — {output['model']}")
        print(f"Metals: {', '.join(output['metals'])}")
        print(f"Pressures: {output['pressures_GPa']} GPa")
        print()
        print("Ranking (by formation energy at highest pressure):")
        print("-" * 80)
        for entry in output["ranking"]:
            ef = entry.get("formation_energy_eV_per_atom", "N/A")
            hull = entry.get("e_above_hull_eV_per_atom_0GPa", "N/A")
            conv = entry.get("converged", "N/A")
            if isinstance(ef, float):
                ef = f"{ef:+.4f}"
            if isinstance(hull, float):
                hull = f"{hull:.4f}"
            print(
                f"  #{entry['rank']:2d}  {entry['formula']:8s}  "
                f"{entry['prototype']:14s}  "
                f"Ef={ef:>8s} eV/atom  "
                f"Ehull(0GPa)={hull:>8s}  "
                f"conv={conv}"
            )
        print()
        print(f"CIF files saved to: {output['output_dir']}")


def _submit_to_slurm(args):
    """No GPU locally — submit this script to SLURM and return job info."""
    import subprocess
    from datetime import datetime, timezone

    script_path = Path(__file__).resolve()
    venv = os.environ.get("VIRTUAL_ENV", "")
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=uma-screen
#SBATCH --partition=venkvis-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --output={out_dir}/slurm-%j.out
#SBATCH --error={out_dir}/slurm-%j.err

source {venv}/bin/activate
export MP_API_KEY="{os.environ.get('MP_API_KEY', '')}"
export HF_TOKEN="{os.environ.get('HF_TOKEN', '')}"

{sys.executable} {script_path} \\
  --metals {args.metals} \\
  --stoichiometries {args.stoichiometries} \\
  --pressures {args.pressures} \\
  --model {args.model} \\
  --device cuda \\
  --fmax {args.fmax} \\
  --steps {args.steps} \\
  --output-dir {out_dir} \\
  --format json
"""
    submit_path = out_dir / "submit.sh"
    submit_path.write_text(slurm_script)
    submit_path.chmod(0o755)

    result = subprocess.run(["sbatch", str(submit_path)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        error = {"status": "error", "error": f"sbatch failed: {result.stderr.strip()}"}
        print(json.dumps(error, indent=2))
        sys.exit(1)

    job_id = None
    for word in result.stdout.strip().split():
        if word.isdigit():
            job_id = word

    output = {
        "status": "SUBMITTED_TO_SLURM",
        "job_id": job_id,
        "output_dir": str(out_dir),
        "note": "No GPU on login node. Job submitted to venkvis-h100. "
                f"Check status: squeue -j {job_id}. "
                f"Results: cat {out_dir}/slurm-{job_id}.out",
        "submit_time": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
