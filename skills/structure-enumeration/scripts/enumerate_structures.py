#!/usr/bin/env python3
"""Generate candidate crystal structures by element substitution.

Takes one or more prototype structures and substitutes the metal site with
a list of target metals. Prototypes can come from Materials Project (by
formula or MP ID) or from local CIF/POSCAR files.
"""

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path.home() / ".scienceclaw" / "enumerated_structures"


def fetch_from_mp(formula: str) -> "Structure":
    """Fetch a structure from Materials Project by formula."""
    from mp_api.client import MPRester
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        print(f"Error: MP_API_KEY required to fetch {formula}", file=sys.stderr)
        sys.exit(1)
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            formula=formula,
            fields=["material_id", "structure", "formula_pretty"],
        )
        if not docs:
            print(f"Warning: no structure found in MP for '{formula}', skipping",
                  file=sys.stderr)
            return None, None
        # Pick the first (lowest energy) entry
        return docs[0].structure, docs[0].material_id


def fetch_by_mpid(mp_id: str) -> "Structure":
    """Fetch a structure from Materials Project by ID."""
    from mp_api.client import MPRester
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        print(f"Error: MP_API_KEY required to fetch {mp_id}", file=sys.stderr)
        sys.exit(1)
    with MPRester(api_key) as mpr:
        structure = mpr.get_structure_by_material_id(mp_id)
        return structure, mp_id


def identify_metal_site(structure) -> str:
    """Identify the metal element (heaviest non-H, non-halogen element)."""
    from pymatgen.core import Element
    elements = set(structure.composition.elements)
    # Exclude H, halogens, noble gases — keep metals/metalloids
    non_h = [e for e in elements if e.symbol != "H"]
    if not non_h:
        print("Error: structure has no non-hydrogen elements", file=sys.stderr)
        sys.exit(1)
    # Pick the heaviest element as the metal site
    return max(non_h, key=lambda e: e.atomic_mass).symbol


def substitute_metal(structure, original_metal: str, new_metal: str):
    """Replace one element with another in a structure."""
    from pymatgen.transformations.standard_transformations import (
        SubstitutionTransformation,
    )
    sub = SubstitutionTransformation({original_metal: new_metal})
    return sub.apply_transformation(structure)


def main():
    parser = argparse.ArgumentParser(
        description="Generate candidate structures by element substitution")
    parser.add_argument("--prototypes",
                        help="Comma-separated formulas to fetch from MP "
                             "(e.g. LaH10,CaH6)")
    parser.add_argument("--mp-ids",
                        help="Comma-separated MP IDs (e.g. mp-1234,mp-5678)")
    parser.add_argument("--prototype-files",
                        help="Comma-separated paths to local CIF/POSCAR files")
    parser.add_argument("--metals", required=True,
                        help="Comma-separated target metals (e.g. Y,Ca,Sc)")
    parser.add_argument("--output-dir", "-o",
                        default=str(DEFAULT_OUTPUT_DIR),
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--format", default="summary",
                        choices=["summary", "json"])
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without generating structures")
    args = parser.parse_args()

    if not args.prototypes and not args.mp_ids and not args.prototype_files:
        print("Error: provide --prototypes, --mp-ids, or --prototype-files",
              file=sys.stderr)
        sys.exit(1)

    metals = [m.strip() for m in args.metals.split(",") if m.strip()]
    output_dir = Path(args.output_dir)

    # Collect prototype structures
    prototypes = []  # list of (structure, source_label, source_metal)

    if args.prototypes:
        for formula in args.prototypes.split(","):
            formula = formula.strip()
            if not formula:
                continue
            print(f"Fetching {formula} from Materials Project...",
                  file=sys.stderr)
            try:
                structure, mp_id = fetch_from_mp(formula)
                if structure is None:
                    continue
                metal = identify_metal_site(structure)
                prototypes.append((structure, formula, metal))
                print(f"  Found {mp_id}: {structure.composition.reduced_formula}, "
                      f"metal site: {metal}, {len(structure)} atoms",
                      file=sys.stderr)
            except Exception as e:
                print(f"  Warning: failed to fetch {formula}: {e}",
                      file=sys.stderr)

    if args.mp_ids:
        for mp_id in args.mp_ids.split(","):
            mp_id = mp_id.strip()
            if not mp_id:
                continue
            print(f"Fetching {mp_id}...", file=sys.stderr)
            try:
                structure, _ = fetch_by_mpid(mp_id)
                metal = identify_metal_site(structure)
                formula = structure.composition.reduced_formula
                prototypes.append((structure, formula, metal))
                print(f"  Got {formula}, metal site: {metal}, "
                      f"{len(structure)} atoms", file=sys.stderr)
            except Exception as e:
                print(f"  Warning: failed to fetch {mp_id}: {e}",
                      file=sys.stderr)

    if args.prototype_files:
        from pymatgen.core import Structure
        for path_str in args.prototype_files.split(","):
            path = Path(path_str.strip())
            if not path.exists():
                print(f"  Warning: file not found: {path}", file=sys.stderr)
                continue
            structure = Structure.from_file(str(path))
            metal = identify_metal_site(structure)
            formula = structure.composition.reduced_formula
            prototypes.append((structure, formula, metal))
            print(f"  Loaded {path.name}: {formula}, metal site: {metal}",
                  file=sys.stderr)

    if not prototypes:
        output = {
            "status": "no_prototypes_found",
            "error": "No prototype structures could be loaded from Materials Project. "
                     "Try different formulas or provide local CIF files via --prototype-files.",
            "attempted_formulas": args.prototypes.split(",") if args.prototypes else [],
        }
        print(json.dumps(output, indent=2))
        return

    # Generate substituted structures
    generated = []
    for structure, proto_formula, original_metal in prototypes:
        for metal in metals:
            if metal == original_metal:
                continue  # skip self-substitution

            label = f"{metal}{proto_formula.replace(original_metal, '').strip()}_{proto_formula}"
            # Cleaner label
            new_formula_approx = proto_formula.replace(original_metal, metal)
            label = f"{new_formula_approx}_from_{proto_formula}"

            entry = {
                "label": label,
                "metal": metal,
                "prototype_formula": proto_formula,
                "original_metal": original_metal,
            }

            if not args.dry_run:
                try:
                    new_struct = substitute_metal(structure, original_metal, metal)
                    cif_path = output_dir / f"{label}.cif"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    new_struct.to(str(cif_path), fmt="cif")
                    entry["formula"] = new_struct.composition.reduced_formula
                    entry["n_atoms"] = len(new_struct)
                    entry["cif_path"] = str(cif_path)
                except Exception as e:
                    entry["error"] = str(e)
                    print(f"  Warning: substitution failed for {label}: {e}",
                          file=sys.stderr)
            else:
                entry["formula"] = new_formula_approx
                entry["cif_path"] = str(output_dir / f"{label}.cif")

            generated.append(entry)

    # Also save the original prototypes
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        for structure, proto_formula, metal in prototypes:
            cif_path = output_dir / f"{proto_formula}_prototype.cif"
            structure.to(str(cif_path), fmt="cif")

    output = {
        "status": "success" if not args.dry_run else "dry_run",
        "output_dir": str(output_dir),
        "prototypes_used": [pf for _, pf, _ in prototypes],
        "metals": metals,
        "total_generated": len(generated),
        "structures": generated,
    }

    if args.format == "json":
        print(json.dumps(output, indent=2))
    else:
        print(f"{'DRY RUN: ' if args.dry_run else ''}Generated "
              f"{len(generated)} structures from "
              f"{len(prototypes)} prototype(s)")
        print(f"  Prototypes: {', '.join(pf for _, pf, _ in prototypes)}")
        print(f"  Metals: {', '.join(metals)}")
        if not args.dry_run:
            print(f"  Output: {output_dir}/")
        print()
        for g in generated:
            err = f" [ERROR: {g['error']}]" if "error" in g else ""
            print(f"  {g['label']:<35s} {g.get('formula', '?'):<12s}"
                  f" {g.get('n_atoms', '?'):>4} atoms{err}")


if __name__ == "__main__":
    main()
