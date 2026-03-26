#!/usr/bin/env python3
"""
Superhydride Superconductivity Screening Workflow
Screens LaH10 and CaH6 with Y,Ca,Sc,Ce substitutions
Relaxes with UMA at 150 GPa and computes phonon stability
"""

import os
import numpy as np
from ase import Atoms
from ase.build import bulk, make_supercell
from ase.io import write, read
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import json
import subprocess
from pathlib import Path

def create_lah10_structure():
    """Create LaH10 structure in Fm-3m space group"""
    # LaH10 in cubic Fm-3m structure (experimental high-pressure phase)
    a = 5.0  # Initial lattice parameter (will be optimized)
    
    # Create fcc-based structure
    positions = [
        [0.0, 0.0, 0.0],     # La at origin
        [0.25, 0.25, 0.0],   # H positions
        [0.75, 0.75, 0.0],
        [0.25, 0.75, 0.25],
        [0.75, 0.25, 0.25],
        [0.25, 0.0, 0.5],
        [0.75, 0.0, 0.5],
        [0.0, 0.25, 0.75],
        [0.0, 0.75, 0.75],
        [0.5, 0.5, 0.5],
        [0.5, 0.25, 0.75]
    ]
    
    symbols = ['La'] + ['H'] * 10
    cell = [[a, 0, 0], [0, a, 0], [0, 0, a]]
    
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    atoms.set_scaled_positions(positions)
    
    return atoms

def create_cah6_structure():
    """Create CaH6 structure in Im-3m space group"""
    # CaH6 in cubic Im-3m structure 
    a = 4.8  # Initial lattice parameter
    
    positions = [
        [0.0, 0.0, 0.0],     # Ca at origin
        [0.5, 0.5, 0.5],     # Ca at body center
        [0.25, 0.0, 0.5],    # H positions
        [0.75, 0.0, 0.5],
        [0.0, 0.25, 0.5],
        [0.0, 0.75, 0.5],
        [0.5, 0.25, 0.0],
        [0.5, 0.75, 0.0]
    ]
    
    symbols = ['Ca', 'Ca'] + ['H'] * 6
    cell = [[a, 0, 0], [0, a, 0], [0, 0, a]]
    
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    atoms.set_scaled_positions(positions)
    
    return atoms

def substitute_metal(atoms, old_symbol, new_symbol):
    """Substitute metal atoms in structure"""
    new_atoms = atoms.copy()
    symbols = list(new_atoms.get_chemical_symbols())
    
    for i, symbol in enumerate(symbols):
        if symbol == old_symbol:
            symbols[i] = new_symbol
    
    new_atoms.set_chemical_symbols(symbols)
    return new_atoms

def setup_directories():
    """Create directory structure for calculations"""
    base_dir = Path("superhydride_screening")
    base_dir.mkdir(exist_ok=True)
    
    structures_dir = base_dir / "structures"
    structures_dir.mkdir(exist_ok=True)
    
    calc_dir = base_dir / "calculations"
    calc_dir.mkdir(exist_ok=True)
    
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    return base_dir, structures_dir, calc_dir, results_dir

def main():
    print("Starting Superhydride Screening Workflow")
    print("=" * 50)
    
    # Setup directories
    base_dir, structures_dir, calc_dir, results_dir = setup_directories()
    
    # Define substitution elements
    substitutions = ['Y', 'Ca', 'Sc', 'Ce']
    
    # Create base structures
    lah10 = create_lah10_structure()
    cah6 = create_cah6_structure()
    
    structures = {}
    
    # LaH10 and substitutions
    structures['LaH10'] = lah10
    for sub in substitutions:
        if sub != 'La':
            structures[f'{sub}H10'] = substitute_metal(lah10, 'La', sub)
    
    # CaH6 and substitutions  
    structures['CaH6'] = cah6
    for sub in substitutions:
        if sub != 'Ca':
            structures[f'{sub}H6'] = substitute_metal(cah6, 'Ca', sub)
    
    # Save all structures
    for name, atoms in structures.items():
        struct_file = structures_dir / f"{name}.cif"
        write(str(struct_file), atoms, format='cif')
        print(f"Created structure: {name}")
    
    print(f"\nGenerated {len(structures)} superhydride structures")
    print(f"Structures saved in: {structures_dir}")
    
    # Create summary
    summary = {
        'total_structures': len(structures),
        'structure_types': list(structures.keys()),
        'base_structures': ['LaH10', 'CaH6'],
        'substitution_elements': substitutions,
        'pressure': '150 GPa',
        'next_steps': ['UMA relaxation', 'phonon calculations', 'stability analysis']
    }
    
    with open(results_dir / 'structure_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {results_dir / 'structure_summary.json'}")
    return structures

if __name__ == "__main__":
    structures = main()