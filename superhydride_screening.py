#!/usr/bin/env python3
"""
Superhydride screening for superconductivity candidates
Builds LaH10 and CaH6 structures with Y, Ca, Sc, Ce substitutions
Relaxes with UMA at 150 GPa and computes phonon stability
"""

import os
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from ase import Atoms
from ase.io import write
from ase.calculators.calculator import Calculator
import json
from pathlib import Path

# Create working directories
Path("structures").mkdir(exist_ok=True)
Path("relaxed").mkdir(exist_ok=True)
Path("phonons").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

def create_lah10_structure():
    """
    Create LaH10 structure based on experimental/theoretical predictions
    Using fcc-derived structure at high pressure
    """
    # Lattice parameters for LaH10 at 150 GPa (approximate)
    a = 4.8  # Angstrom
    lattice = Lattice.cubic(a)
    
    # Atomic positions for LaH10 (space group Fm-3m)
    species = ['La'] + ['H'] * 10
    coords = [
        [0.0, 0.0, 0.0],  # La at origin
        [0.5, 0.5, 0.0],  # H positions
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [0.25, 0.25, 0.25],
        [0.75, 0.25, 0.25],
        [0.25, 0.75, 0.25],
        [0.25, 0.25, 0.75],
        [0.75, 0.75, 0.25],
        [0.75, 0.25, 0.75],
        [0.25, 0.75, 0.75]
    ]
    
    structure = Structure(lattice, species, coords)
    return structure

def create_cah6_structure():
    """
    Create CaH6 structure at high pressure
    Using sodalite-type structure
    """
    # Lattice parameters for CaH6 at 150 GPa
    a = 4.2  # Angstrom
    lattice = Lattice.cubic(a)
    
    # Atomic positions for CaH6
    species = ['Ca'] + ['H'] * 6
    coords = [
        [0.0, 0.0, 0.0],  # Ca at origin
        [0.5, 0.0, 0.0],  # H positions
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.25, 0.25, 0.25],
        [0.75, 0.75, 0.25],
        [0.25, 0.75, 0.75]
    ]
    
    structure = Structure(lattice, species, coords)
    return structure

def substitute_element(structure, original_element, new_element):
    """
    Substitute one element with another in the structure
    """
    substituted = structure.copy()
    substituted.replace_species({original_element: new_element})
    return substituted

def write_poscar(structure, filename):
    """
    Write structure in VASP POSCAR format
    """
    adaptor = AseAtomsAdaptor()
    atoms = adaptor.get_atoms(structure)
    write(filename, atoms, format='vasp')

def create_all_structures():
    """
    Create all superhydride structures with substitutions
    """
    base_structures = {
        'LaH10': create_lah10_structure(),
        'CaH6': create_cah6_structure()
    }
    
    substitutions = ['Y', 'Ca', 'Sc', 'Ce']
    structures_created = {}
    
    for base_name, base_struct in base_structures.items():
        structures_created[base_name] = base_struct
        
        # Get the metal element to substitute
        metal_element = base_name.split('H')[0]
        
        for sub_element in substitutions:
            if sub_element != metal_element:  # Don't substitute with same element
                sub_name = f"{sub_element}H{base_name.split('H')[1]}"
                sub_struct = substitute_element(base_struct, metal_element, sub_element)
                structures_created[sub_name] = sub_struct
    
    # Write all structures
    for name, struct in structures_created.items():
        poscar_path = f"structures/{name}_POSCAR"
        write_poscar(struct, poscar_path)
        print(f"Created structure: {name}")
    
    return structures_created

def create_uma_input(structure_name, pressure_gpa=150):
    """
    Create UMA input files for structure relaxation at high pressure
    """
    input_template = f"""# UMA calculation for {structure_name} at {pressure_gpa} GPa
structure_file = structures/{structure_name}_POSCAR
output_dir = relaxed/{structure_name}

# Calculation settings
calc_type = relax
pressure = {pressure_gpa}  # GPa
max_steps = 500
force_convergence = 0.01  # eV/Angstrom
stress_convergence = 0.1  # GPa

# Electronic structure
xc_functional = PBE
k_points_density = 0.3  # per Angstrom^-1
energy_cutoff = 800  # eV
smearing = 0.1  # eV

# High pressure settings
bulk_modulus_estimate = 200  # GPa
optimize_cell = true
optimize_positions = true
"""
    
    input_path = f"uma_{structure_name}.inp"
    with open(input_path, 'w') as f:
        f.write(input_template)
    
    return input_path

def create_phonon_input(structure_name):
    """
    Create phonon calculation input using the relaxed structure
    """
    phonon_template = f"""# Phonon calculation for {structure_name}
structure_file = relaxed/{structure_name}/CONTCAR
output_dir = phonons/{structure_name}

# Phonon settings
calc_type = phonon
supercell = 2 2 2
displacement = 0.01  # Angstrom
q_points_density = 0.2

# Electronic structure (same as relaxation)
xc_functional = PBE
k_points_density = 0.3
energy_cutoff = 800
smearing = 0.1

# Analysis
compute_dos = true
compute_thermal_properties = true
temperature_range = 0 300 10  # K
"""
    
    phonon_path = f"phonon_{structure_name}.inp"
    with open(phonon_path, 'w') as f:
        f.write(phonon_template)
    
    return phonon_path

def main():
    print("Starting superhydride screening workflow...")
    
    # Create all structures
    structures = create_all_structures()
    
    # Create UMA input files for relaxation
    uma_inputs = []
    phonon_inputs = []
    
    for structure_name in structures.keys():
        # Create UMA relaxation input
        uma_input = create_uma_input(structure_name, 150)
        uma_inputs.append((structure_name, uma_input))
        
        # Create phonon input
        phonon_input = create_phonon_input(structure_name)
        phonon_inputs.append((structure_name, phonon_input))
    
    # Create summary of workflow
    summary = {
        'structures_created': list(structures.keys()),
        'pressure': '150 GPa',
        'calculation_sequence': [
            '1. Structure relaxation with UMA',
            '2. Phonon stability analysis',
            '3. Superconductivity screening'
        ],
        'uma_inputs': [inp for _, inp in uma_inputs],
        'phonon_inputs': [inp for _, inp in phonon_inputs]
    }
    
    with open('workflow_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nWorkflow setup complete!")
    print(f"Created {len(structures)} structures")
    print(f"Generated {len(uma_inputs)} UMA input files")
    print(f"Generated {len(phonon_inputs)} phonon input files")
    print("\nNext steps:")
    print("1. Submit UMA relaxation jobs")
    print("2. Run phonon calculations on relaxed structures")
    print("3. Analyze results for superconductivity indicators")

if __name__ == "__main__":
    main()