#!/usr/bin/env python3
"""Enumerate candidate MHx structures from known superconducting hydride prototypes.

Takes known prototype crystal structures (H3S Im-3m, LaH10 Fm-3m, CaH6 Im-3m)
and substitutes the metal site with a set of target metals to generate candidate
structures for screening.

Requires: pymatgen
"""

import argparse
import json
import sys
from pathlib import Path

# Known superconducting hydride prototypes with Wyckoff positions.
# References:
#   H3S (Im-3m):   Drozdov et al., Nature 525, 73 (2015)
#   CaH6 (Im-3m):  Wang et al., PNAS 109, 6463 (2012)
#   YH9 (P63/mmc): Peng et al., PRL 119, 107001 (2017)
#   LaH10 (Fm-3m): Drozdov et al., Nature 569, 528 (2019)
#   MH12 (Im-3m):  Hypothetical, BCC metal + 24h H sites
PROTOTYPES = {
    "H3S": {
        "spacegroup": 229,  # Im-3m
        "lattice_a": 3.09,
        "species": ["S", "H"],
        "coords": [
            [0.0, 0.0, 0.0],      # 2a: metal/S site
            [0.5, 0.0, 0.5],      # 6b: H site
        ],
        "metal_index": 0,
        "H_per_M": 3,
        "description": "BCC-based, Tc~200K at 150 GPa",
    },
    "CaH6": {
        "spacegroup": 229,  # Im-3m
        "lattice_a": 3.54,
        "species": ["Ca", "H"],
        "coords": [
            [0.0, 0.0, 0.0],      # 2a: metal site
            [0.25, 0.0, 0.5],     # 12d: H site (sodalite cage)
        ],
        "metal_index": 0,
        "H_per_M": 6,
        "description": "Sodalite-like H cage, Tc~220K at 150 GPa",
    },
    "YH9": {
        "spacegroup": 194,  # P63/mmc
        "lattice_a": 3.60,
        "lattice_c": 5.50,
        "species": ["Y", "H", "H"],
        "coords": [
            [0.0, 0.0, 0.25],         # 2b: metal site
            [0.167, 0.333, 0.25],      # 6h: H equatorial
            [0.167, 0.333, 0.583],     # 12k: H axial
        ],
        "metal_index": 0,
        "H_per_M": 9,
        "description": "Hexagonal, Tc~240K at 200 GPa (predicted)",
    },
    "LaH10": {
        "spacegroup": 225,  # Fm-3m
        "lattice_a": 5.10,
        "species": ["La", "H", "H"],
        "coords": [
            [0.0, 0.0, 0.0],         # 4a: metal site
            [0.25, 0.25, 0.25],       # 8c: H site (tetrahedral)
            [0.118, 0.118, 0.118],    # 32f: H site (clathrate cage)
        ],
        "metal_index": 0,
        "H_per_M": 10,
        "description": "FCC clathrate, Tc~250K at 170 GPa",
    },
    "MH12": {
        "spacegroup": 229,  # Im-3m
        "lattice_a": 3.80,
        "species": ["La", "H", "H"],
        "coords": [
            [0.0, 0.0, 0.0],       # 2a: metal site
            [0.25, 0.0, 0.5],      # 12d: H site (sodalite cage)
            [0.18, 0.18, 0.0],     # 12e: H site (additional cage)
        ],
        "metal_index": 0,
        "H_per_M": 12,
        "description": "Hypothetical hyper-hydride, BCC-based with extra H",
    },
}

# Default metals known to form superconducting hydrides
DEFAULT_METALS = ["La", "Y", "Ca", "Th", "Ce", "Sc", "Lu", "Ba"]


def build_candidate(prototype_name: str, metal: str) -> dict:
    """Build a candidate structure by substituting the metal site.

    Returns dict with structure info and pymatgen Structure object.
    """
    from pymatgen.core import Structure, Lattice

    proto = PROTOTYPES[prototype_name]
    species = list(proto["species"])
    species[proto["metal_index"]] = metal

    if "lattice_c" in proto:
        # Hexagonal lattice
        lattice = Lattice.hexagonal(proto["lattice_a"], proto["lattice_c"])
    else:
        lattice = Lattice.cubic(proto["lattice_a"])

    structure = Structure.from_spacegroup(
        proto["spacegroup"],
        lattice,
        species,
        proto["coords"],
    )

    formula = structure.composition.reduced_formula
    sg_symbol = {229: "Im-3m", 225: "Fm-3m", 194: "P63mmc"}.get(
        proto["spacegroup"], str(proto["spacegroup"]))
    label = f"{metal}H{proto['H_per_M']}_{sg_symbol}"

    return {
        "label": label,
        "formula": formula,
        "prototype": prototype_name,
        "metal": metal,
        "spacegroup": proto["spacegroup"],
        "spacegroup_symbol": sg_symbol,
        "H_per_M": proto["H_per_M"],
        "n_atoms": len(structure),
        "lattice_a": proto["lattice_a"],
        "structure": structure,
    }


def enumerate_candidates(metals: list[str],
                         prototypes: list[str]) -> list[dict]:
    """Generate all MHx candidates from the given metals and prototypes."""
    candidates = []
    for proto_name in prototypes:
        if proto_name not in PROTOTYPES:
            print(f"Warning: unknown prototype '{proto_name}', skipping",
                  file=sys.stderr)
            continue
        for metal in metals:
            try:
                candidate = build_candidate(proto_name, metal)
                candidates.append(candidate)
            except Exception as e:
                print(f"Warning: failed to build {metal} in {proto_name}: {e}",
                      file=sys.stderr)
    return candidates


def save_candidates(candidates: list[dict], output_dir: Path) -> list[dict]:
    """Write each candidate as a CIF file. Returns updated candidate list."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for c in candidates:
        cif_path = output_dir / f"{c['label']}.cif"
        c["structure"].to(str(cif_path), fmt="cif")
        info = {k: v for k, v in c.items() if k != "structure"}
        info["cif_path"] = str(cif_path)
        results.append(info)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Enumerate candidate MHx hydride structures from prototypes")
    parser.add_argument("--metals", default=",".join(DEFAULT_METALS),
                        help=f"Comma-separated metals (default: {','.join(DEFAULT_METALS)})")
    parser.add_argument("--prototypes", default="H3S,CaH6,YH9,LaH10,MH12",
                        help="Comma-separated prototype names (default: H3S,CaH6,YH9,LaH10,MH12)")
    parser.add_argument("--output-dir", "-o", default="./hydride_candidates",
                        help="Directory to write CIF files")
    parser.add_argument("--format", default="summary",
                        choices=["summary", "json"])
    parser.add_argument("--dry-run", action="store_true",
                        help="List candidates without writing files")
    args = parser.parse_args()

    metals = [m.strip() for m in args.metals.split(",") if m.strip()]
    prototypes = [p.strip() for p in args.prototypes.split(",") if p.strip()]
    output_dir = Path(args.output_dir)

    candidates = enumerate_candidates(metals, prototypes)

    if args.dry_run:
        results = []
        for c in candidates:
            info = {k: v for k, v in c.items() if k != "structure"}
            info["cif_path"] = str(output_dir / f"{c['label']}.cif")
            results.append(info)
    else:
        results = save_candidates(candidates, output_dir)

    output = {
        "dry_run": args.dry_run,
        "metals": metals,
        "prototypes": prototypes,
        "total": len(results),
        "candidates": results,
    }

    if args.format == "json":
        print(json.dumps(output, indent=2))
    else:
        print(f"Enumerated {len(results)} candidate structures")
        print(f"  Metals: {', '.join(metals)}")
        print(f"  Prototypes: {', '.join(prototypes)}")
        if not args.dry_run:
            print(f"  Output: {output_dir}/")
        print()
        for r in results:
            print(f"  {r['label']:25s}  {r['formula']:10s}  "
                  f"SG {r['spacegroup']}  {r['n_atoms']:3d} atoms  "
                  f"a={r['lattice_a']:.2f} A")


if __name__ == "__main__":
    main()
