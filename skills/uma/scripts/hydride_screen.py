#!/usr/bin/env python3
"""Screen hydride superconductor candidates using UMA relaxation.

End-to-end pipeline:
  1. Enumerate MHx structures from known prototypes
  2. Relax each at multiple pressures using UMA
  3. Save relaxed structures as CIF files
  4. At 0 GPa: compute energy above convex hull using Materials Project data
  5. At high pressure: report formation energies (no hull data available)
  6. Rank candidates by stability

Can run locally (GPU required) or submit to SLURM via --slurm.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ground-state crystal structures for elemental references
# (structure_type, lattice_a in Angstrom)
ELEMENTAL_STRUCTURES = {
    "La": ("fcc", 5.31),
    "Y":  ("hcp", 3.65),
    "Ca": ("fcc", 5.58),
    "Th": ("fcc", 5.08),
    "Ce": ("fcc", 5.16),
    "Sc": ("hcp", 3.31),
    "Lu": ("hcp", 3.50),
    "Ba": ("bcc", 5.02),
    "S":  ("orthorhombic", 10.47),
}


def build_elemental_atoms(element: str):
    """Build ASE Atoms for an elemental reference structure."""
    from ase.build import bulk
    info = ELEMENTAL_STRUCTURES.get(element)
    if info is None:
        return bulk(element)
    struct_type, a = info
    if struct_type == "fcc":
        return bulk(element, "fcc", a=a)
    elif struct_type == "bcc":
        return bulk(element, "bcc", a=a)
    elif struct_type == "hcp":
        return bulk(element, "hcp", a=a, c=a * 1.633)
    else:
        return bulk(element)


def build_h2_reference():
    """Build an H2 molecule in a box for the hydrogen reference."""
    from ase import Atoms
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]],
                 cell=[10, 10, 10], pbc=True)


def compute_ehull_mp(candidates_at_0gpa: list, uma_ref_energies: dict) -> dict:
    """Compute energy above hull at 0 GPa using Materials Project phase diagrams.

    For each M-H binary system, fetches all known MP entries and constructs
    a convex hull. UMA-relaxed candidates are added as new entries and their
    distance above the hull is computed.

    Returns dict mapping candidate label -> ehull (eV/atom).
    """
    try:
        from mp_api.client import MPRester
        from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
        from pymatgen.core import Composition
    except ImportError:
        print("Warning: mp-api/pymatgen not available for hull calculation",
              file=sys.stderr)
        return {}

    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        print("Warning: MP_API_KEY not set, skipping convex hull analysis",
              file=sys.stderr)
        return {}

    # Group candidates by metal
    by_metal = {}
    for c in candidates_at_0gpa:
        metal = c["metal"]
        if metal not in by_metal:
            by_metal[metal] = []
        by_metal[metal].append(c)

    ehull_results = {}

    with MPRester(api_key) as mpr:
        for metal, cands in by_metal.items():
            chemsys = f"{metal}-H"
            print(f"  Fetching MP entries for {chemsys}...", file=sys.stderr)

            try:
                mp_entries = mpr.get_entries_in_chemsys(
                    chemsys,
                    additional_criteria={"thermo_types": ["GGA_GGA+U"]},
                )
            except Exception as e:
                print(f"  Warning: failed to fetch {chemsys}: {e}",
                      file=sys.stderr)
                # Fall back to formation energy
                for c in cands:
                    ehull_results[c["label"]] = None
                continue

            if not mp_entries:
                print(f"  Warning: no MP entries for {chemsys}", file=sys.stderr)
                for c in cands:
                    ehull_results[c["label"]] = None
                continue

            # Add UMA-computed entries for our candidates
            uma_entries = []
            for c in cands:
                comp = Composition(c["formula"])
                # Total energy (not per atom)
                total_energy = c["energy_per_atom_eV"] * c["n_atoms"]
                entry = PDEntry(comp, total_energy, name=c["label"])
                uma_entries.append(entry)

            # Build phase diagram with MP + UMA entries
            all_entries = list(mp_entries) + uma_entries
            try:
                pd = PhaseDiagram(all_entries)
                for entry in uma_entries:
                    ehull = pd.get_e_above_hull(entry)
                    label = entry.name
                    ehull_results[label] = round(float(ehull), 6)
                    print(f"  {label}: E_hull = {ehull:.4f} eV/atom",
                          file=sys.stderr)
            except Exception as e:
                print(f"  Warning: phase diagram failed for {chemsys}: {e}",
                      file=sys.stderr)
                for c in cands:
                    ehull_results[c["label"]] = None

    return ehull_results


def run_screening(metals, prototypes, pressures, model, task, device,
                  fmax, max_steps, output_dir, dry_run=False):
    """Run the full screening pipeline."""
    from hydride_enumerate import enumerate_candidates, save_candidates

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    relaxed_dir = output_dir / "relaxed_structures"
    relaxed_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Enumerate candidates
    print("Step 1: Enumerating candidate structures...", file=sys.stderr)
    candidates = enumerate_candidates(metals, prototypes)
    cif_dir = output_dir / "initial_structures"
    saved = save_candidates(candidates, cif_dir)
    print(f"  Generated {len(saved)} candidates", file=sys.stderr)

    if dry_run:
        return {
            "dry_run": True,
            "candidates": saved,
            "pressures_GPa": pressures,
            "total_calculations": len(saved) * len(pressures),
            "elemental_references": list(set(
                c["metal"] for c in saved
            )) + ["H"],
            "model": model,
            "task": task,
        }

    # Step 2: Load UMA model
    print(f"Step 2: Loading UMA model ({model})...", file=sys.stderr)
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
    predictor = pretrained_mlip.get_predict_unit(model, device=device)

    def relax_atoms(atoms, pressure_gpa, label="", save_cif_path=None):
        """Relax atoms and return energy results. Optionally save CIF."""
        from ase.optimize import FIRE
        from ase.filters import FrechetCellFilter
        from ase.io import write as ase_write
        import numpy as np

        calc = FAIRChemCalculator(predictor, task_name=task)
        atoms.calc = calc

        pressure_ev = pressure_gpa / 160.21766208
        opt_atoms = FrechetCellFilter(atoms, scalar_pressure=pressure_ev)
        opt = FIRE(opt_atoms, logfile=None)
        converged = opt.run(fmax=fmax, steps=max_steps)

        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        fmax_val = float(np.max(np.linalg.norm(forces, axis=1)))
        volume = atoms.get_volume()
        cell = atoms.get_cell()
        lengths = cell.lengths()

        if save_cif_path:
            ase_write(str(save_cif_path), atoms, format="cif")

        return {
            "energy_eV": round(float(energy), 6),
            "energy_per_atom_eV": round(float(energy) / len(atoms), 6),
            "n_atoms": len(atoms),
            "converged": bool(converged),
            "steps": opt.nsteps,
            "fmax_achieved": round(fmax_val, 6),
            "volume_A3": round(float(volume), 4),
            "volume_per_atom_A3": round(float(volume) / len(atoms), 4),
            "lattice_a": round(float(lengths[0]), 4),
            "lattice_b": round(float(lengths[1]), 4),
            "lattice_c": round(float(lengths[2]), 4),
        }

    # Step 3: Compute elemental reference energies
    print("Step 3: Computing elemental reference energies...", file=sys.stderr)
    unique_metals = list(set(c["metal"] for c in saved))
    ref_energies = {}

    for pressure in pressures:
        for metal in unique_metals:
            label = f"{metal}_{pressure}GPa"
            print(f"  Relaxing {label}...", file=sys.stderr)
            atoms = build_elemental_atoms(metal)
            result = relax_atoms(atoms, pressure, label)
            ref_energies[label] = result
            print(f"    E/atom = {result['energy_per_atom_eV']:.4f} eV",
                  file=sys.stderr)

        label = f"H_{pressure}GPa"
        print(f"  Relaxing {label}...", file=sys.stderr)
        h2 = build_h2_reference()
        result = relax_atoms(h2, pressure, label)
        ref_energies[label] = result
        print(f"    E/atom = {result['energy_per_atom_eV']:.4f} eV",
              file=sys.stderr)

    # Step 4: Relax candidates and compute formation energies
    print("Step 4: Relaxing candidates...", file=sys.stderr)
    screening_results = []
    calc_num = 0
    total_calcs = len(saved) * len(pressures)

    for cand in saved:
        for pressure in pressures:
            calc_num += 1
            label = f"{cand['label']}_{pressure}GPa"
            print(f"  [{calc_num}/{total_calcs}] {label}...", file=sys.stderr)

            from ase.io import read as ase_read
            atoms = ase_read(cand["cif_path"])

            # Save relaxed structure
            cif_out = relaxed_dir / f"{cand['label']}_{pressure}GPa.cif"
            result = relax_atoms(atoms, pressure, label, save_cif_path=cif_out)

            # Formation energy
            metal = cand["metal"]
            h_per_m = cand["H_per_M"]
            e_metal_ref = ref_energies[f"{metal}_{pressure}GPa"]["energy_per_atom_eV"]
            e_h_ref = ref_energies[f"H_{pressure}GPa"]["energy_per_atom_eV"]
            e_compound = result["energy_per_atom_eV"]

            frac_metal = 1.0 / (1.0 + h_per_m)
            frac_h = h_per_m / (1.0 + h_per_m)
            e_ref_mix = frac_metal * e_metal_ref + frac_h * e_h_ref
            formation_energy = e_compound - e_ref_mix

            entry = {
                "label": cand["label"],
                "formula": cand["formula"],
                "prototype": cand["prototype"],
                "metal": metal,
                "H_per_M": h_per_m,
                "pressure_GPa": pressure,
                **result,
                "formation_energy_eV_per_atom": round(formation_energy, 6),
                "relaxed_cif": str(cif_out),
            }
            screening_results.append(entry)

            status = "STABLE" if formation_energy < 0 else "unstable"
            print(f"    E_f = {formation_energy:.4f} eV/atom ({status})",
                  file=sys.stderr)

    # Step 5: Convex hull analysis at 0 GPa
    results_0gpa = [r for r in screening_results if r["pressure_GPa"] == 0.0]
    ehull_map = {}
    if results_0gpa:
        print("\nStep 5: Computing energy above convex hull (0 GPa)...",
              file=sys.stderr)
        ehull_map = compute_ehull_mp(results_0gpa, ref_energies)

    # Add ehull to results
    for r in screening_results:
        if r["pressure_GPa"] == 0.0:
            r["e_above_hull_eV"] = ehull_map.get(r["label"])
        else:
            r["e_above_hull_eV"] = None  # No hull data at high pressure

    # Sort: 0 GPa by ehull (if available), then by formation energy
    def sort_key(r):
        ehull = r.get("e_above_hull_eV")
        if ehull is not None:
            return (0, ehull)  # Prioritize hull results
        return (1, r["formation_energy_eV_per_atom"])

    screening_results.sort(key=sort_key)

    return {
        "screening_results": screening_results,
        "elemental_references": ref_energies,
        "parameters": {
            "model": model,
            "task": task,
            "pressures_GPa": pressures,
            "fmax": fmax,
            "max_steps": max_steps,
            "metals": metals,
            "prototypes": prototypes,
        },
        "summary": {
            "total_candidates": len(saved),
            "total_calculations": len(screening_results),
            "stable_by_formation": sum(1 for r in screening_results
                                       if r["formation_energy_eV_per_atom"] < 0),
            "on_hull_count": sum(1 for r in screening_results
                                 if r.get("e_above_hull_eV") is not None
                                 and r["e_above_hull_eV"] < 0.025),
            "most_stable": screening_results[0]["label"] if screening_results else None,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Screen hydride superconductor candidates with UMA")
    parser.add_argument("--metals", default="La,Y,Ca,Th,Ce,Sc,Lu,Ba",
                        help="Comma-separated metals")
    parser.add_argument("--prototypes", default="H3S,CaH6,YH9,LaH10,MH12",
                        help="Comma-separated prototype names")
    parser.add_argument("--pressures", default="0,150",
                        help="Comma-separated pressures in GPa")
    parser.add_argument("--model", "-m", default="uma-m-1p1",
                        choices=["uma-s-1p1", "uma-s-1p2", "uma-m-1p1"])
    parser.add_argument("--task", "-t", default="omat",
                        choices=["omat", "omol", "omc", "oc20", "odac"])
    parser.add_argument("--device", default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=500,
                        help="Max optimizer steps per relaxation (default: 500)")
    parser.add_argument("--output-dir", "-o",
                        default="./hydride_screening",
                        help="Directory for output files")
    parser.add_argument("--slurm", action="store_true",
                        help="Submit as SLURM job instead of running locally")
    parser.add_argument("--partition", default="venkvis-h100")
    parser.add_argument("--walltime", default="04:00:00")
    parser.add_argument("--dry-run", action="store_true",
                        help="Enumerate candidates only, skip relaxation")
    parser.add_argument("--format", default="summary",
                        choices=["summary", "json"])
    args = parser.parse_args()

    metals = [m.strip() for m in args.metals.split(",") if m.strip()]
    prototypes = [p.strip() for p in args.prototypes.split(",") if p.strip()]
    pressures = [float(p) for p in args.pressures.split(",")]
    output_dir = Path(args.output_dir)

    if args.slurm:
        _submit_slurm(args, metals, prototypes, pressures, output_dir)
        return

    results = run_screening(
        metals=metals,
        prototypes=prototypes,
        pressures=pressures,
        model=args.model,
        task=args.task,
        device=args.device,
        fmax=args.fmax,
        max_steps=args.steps,
        output_dir=str(output_dir),
        dry_run=args.dry_run,
    )

    # Save results JSON
    if not args.dry_run:
        results_path = output_dir / "screening_results.json"
        results_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {results_path}", file=sys.stderr)

    if args.format == "json":
        print(json.dumps(results, indent=2))
    else:
        _print_summary(results, args.dry_run)


def _print_summary(results, dry_run):
    """Print human-readable summary."""
    if dry_run:
        print(f"=== DRY RUN: Hydride Screening Plan ===")
        print(f"  Candidates: {len(results['candidates'])}")
        print(f"  Pressures:  {results['pressures_GPa']} GPa")
        print(f"  Total calculations: {results['total_calculations']}")
        print(f"  Model: {results['model']}")
        print()
        print(f"  {'Label':<25s} {'Formula':<10s} {'Prototype':<10s} {'Atoms':>5s}")
        print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*5}")
        for c in results["candidates"]:
            print(f"  {c['label']:<25s} {c['formula']:<10s} "
                  f"{c['prototype']:<10s} {c['n_atoms']:>5d}")
        return

    s = results["summary"]
    print(f"\n=== Hydride Screening Results ===")
    print(f"  Model: {results['parameters']['model']}")
    print(f"  Candidates: {s['total_candidates']}")
    print(f"  Calculations: {s['total_calculations']}")
    print(f"  Stable (E_f < 0): {s['stable_by_formation']}")
    print(f"  On/near hull (<25 meV): {s['on_hull_count']}")
    print()
    print(f"  {'Label':<25s} {'P(GPa)':>7s} {'E/atom':>10s} "
          f"{'E_f':>10s} {'E_hull':>10s} {'V/atom':>8s} {'Conv':>5s}")
    print(f"  {'-'*25} {'-'*7} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*5}")
    for r in results["screening_results"]:
        ehull_str = f"{r['e_above_hull_eV']:>10.4f}" if r.get("e_above_hull_eV") is not None else "       N/A"
        marker = "*" if (r.get("e_above_hull_eV") is not None and r["e_above_hull_eV"] < 0.025) \
                     or (r.get("e_above_hull_eV") is None and r["formation_energy_eV_per_atom"] < 0) \
                 else " "
        print(f"{marker} {r['label']:<25s} {r['pressure_GPa']:>7.0f} "
              f"{r['energy_per_atom_eV']:>10.4f} "
              f"{r['formation_energy_eV_per_atom']:>10.4f} "
              f"{ehull_str} "
              f"{r['volume_per_atom_A3']:>8.2f} "
              f"{'yes' if r['converged'] else 'NO':>5s}")


def _submit_slurm(args, metals, prototypes, pressures, output_dir):
    """Generate and submit a SLURM script for the screening."""
    import subprocess
    from datetime import datetime, timezone

    output_dir.mkdir(parents=True, exist_ok=True)
    venv = os.environ.get("VIRTUAL_ENV", "")
    script_path = Path(__file__).resolve()

    cmd_parts = [
        sys.executable, str(script_path),
        "--metals", ",".join(metals),
        "--prototypes", ",".join(prototypes),
        "--pressures", ",".join(str(p) for p in pressures),
        "--model", args.model,
        "--task", args.task,
        "--device", "cuda",
        "--fmax", str(args.fmax),
        "--steps", str(args.steps),
        "--output-dir", str(output_dir),
        "--format", "json",
    ]

    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=hydride-screen
#SBATCH --partition={args.partition}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time={args.walltime}
#SBATCH --output={output_dir}/slurm-%j.out
#SBATCH --error={output_dir}/slurm-%j.err
#SBATCH --gres=gpu:1

source {venv}/bin/activate
export MP_API_KEY="{os.environ.get('MP_API_KEY', '')}"
export HF_TOKEN="{os.environ.get('HF_TOKEN', '')}"

echo "=== Hydride Screening ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

{' '.join(cmd_parts)} | tee {output_dir}/screening_results.json

echo "End: $(date)"
"""
    submit_path = output_dir / "submit.sh"
    submit_path.write_text(slurm_script)
    submit_path.chmod(0o755)

    if args.dry_run:
        print(json.dumps({
            "dry_run": True,
            "slurm_script": slurm_script,
            "output_dir": str(output_dir),
        }, indent=2))
        return

    if os.environ.get("SLURM_JOB_ID"):
        print("Error: refusing to submit from inside a compute node",
              file=sys.stderr)
        sys.exit(1)

    result = subprocess.run(
        ["sbatch", str(submit_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Error: sbatch failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    job_id = None
    for word in result.stdout.strip().split():
        if word.isdigit():
            job_id = word
            break

    out = {
        "job_id": job_id,
        "output_dir": str(output_dir),
        "status": "PENDING",
        "submit_time": datetime.now(timezone.utc).isoformat(),
    }
    if args.format == "json":
        print(json.dumps(out, indent=2))
    else:
        for k, v in out.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
