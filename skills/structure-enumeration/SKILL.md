---
name: structure-enumeration
description: Generate candidate crystal structures by element substitution in prototype structures
metadata:
---

# Structure Enumeration Skill

Generate candidate crystal structures by substituting elements in a prototype structure. This is a general-purpose tool for combinatorial materials screening — not specific to any material class.

## Scripts

### `enumerate_structures.py` — Substitute elements in a prototype

From a Materials Project formula:
```bash
python3 {baseDir}/scripts/enumerate_structures.py \
  --prototypes LaH10,CaH6 \
  --metals Y,Ca,Sc,Ce,Ba \
  --output-dir ~/.scienceclaw/enumerated_structures \
  --format json
```

From a local CIF file:
```bash
python3 {baseDir}/scripts/enumerate_structures.py \
  --prototype-files LaH10.cif,CaH6.cif \
  --metals Y,Sc \
  --output-dir ./candidates \
  --format json
```

From a Materials Project ID:
```bash
python3 {baseDir}/scripts/enumerate_structures.py \
  --mp-ids mp-1234,mp-5678 \
  --metals Y,Ca \
  --format json
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `--prototypes` | Comma-separated chemical formulas to fetch from Materials Project (e.g. `LaH10,CaH6`) |
| `--mp-ids` | Comma-separated Materials Project IDs (e.g. `mp-1234,mp-5678`) |
| `--prototype-files` | Comma-separated paths to local CIF/POSCAR files |
| `--metals` | Comma-separated target metals to substitute (e.g. `Y,Ca,Sc,Ce`) |
| `--output-dir` | Directory for output CIF files (default: `~/.scienceclaw/enumerated_structures`) |
| `--format` | `summary` \| `json` |
| `--dry-run` | Show plan without generating structures |

## How It Works

1. Loads each prototype structure (from MP or local file)
2. Identifies the metal site (heaviest non-hydrogen element)
3. For each target metal, substitutes the metal site and writes a new CIF file
4. Reports all generated structures as JSON

The original prototype metal is skipped (no self-substitution).

## Output (JSON)

```json
{
  "status": "success",
  "output_dir": "/home/user/.scienceclaw/enumerated_structures",
  "prototypes_used": ["LaH10", "CaH6"],
  "metals": ["Y", "Ca", "Sc"],
  "total_generated": 6,
  "structures": [
    {
      "label": "YH10_from_LaH10",
      "formula": "YH10",
      "prototype_formula": "LaH10",
      "metal": "Y",
      "n_atoms": 44,
      "cif_path": "/home/user/.scienceclaw/enumerated_structures/YH10_from_LaH10.cif"
    }
  ]
}
```
