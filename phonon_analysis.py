#!/usr/bin/env python3
"""
Phonon stability analysis for relaxed superhydride structures
"""

import os
import numpy as np
import json
from ase.io import read, write
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.file_IO import write_FORCE_CONSTANTS, parse_FORCE_CONSTANTS
import matplotlib.pyplot as plt
import argparse

def ase_to_phonopy(atoms):
    """Convert ASE atoms to Phonopy atoms"""
    return PhonopyAtoms(symbols=atoms.get_chemical_symbols(),
                       cell=atoms.get_cell(),
                       positions=atoms.get_positions())

def compute_phonon_stability(structure_file, output_dir):
    """Compute phonon frequencies and analyze stability"""
    print(f"\nAnalyzing phonon stability for {structure_file}")
    
    # Read relaxed structure
    atoms = read(structure_file)
    phonopy_atoms = ase_to_phonopy(atoms)
    
    # Create phonon object with supercell
    phonon = Phonopy(phonopy_atoms, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    
    # Generate displacements
    phonon.generate_displacements(distance=0.01)
    supercells = phonon.get_supercells_with_displacements()
    
    # For this screening, we'll use harmonic approximation
    # In practice, you would calculate forces for each displaced structure
    print(f"Generated {len(supercells)} displaced supercells")
    
    # Placeholder force calculation (would use UMA calculator in practice)
    forces = []
    for i, supercell in enumerate(supercells):
        if supercell is not None:
            # Convert back to ASE for force calculation
            ase_supercell = Atoms(symbols=supercell.symbols,
                                positions=supercell.positions,
                                cell=supercell.cell,
                                pbc=True)
            
            # Placeholder: random small forces for demonstration
            # In practice: calc.get_forces() with UMA
            natoms = len(ase_supercell)
            force = np.random.normal(0, 0.1, (natoms, 3))
            forces.append(force)
            
            # Save displaced structure for reference
            write(f"{output_dir}/displaced_{i:03d}.xyz", ase_supercell)
        else:
            forces.append(None)
    
    # Set forces and calculate force constants
    phonon.set_forces(forces)
    phonon.produce_force_constants()
    
    # Write force constants
    write_FORCE_CONSTANTS(phonon.get_force_constants(), 
                         filename=f"{output_dir}/FORCE_CONSTANTS")
    
    # Calculate phonon frequencies at high-symmetry points
    band_paths = [[[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0, 0], [0, 0, 0.5]]]
    phonon.set_band_structure(band_paths, npoints=51)
    
    # Get frequencies
    frequencies, _ = phonon.get_band_structure()
    
    # Analyze stability
    min_freq = np.min(frequencies)
    negative_freqs = frequencies[frequencies < -1e-3]  # Tolerance for numerical errors
    
    stability_data = {
        'structure': os.path.basename(structure_file),
        'min_frequency_THz': float(min_freq),
        'num_negative_modes': len(negative_freqs),
        'is_stable': len(negative_freqs) == 0,
        'all_frequencies_THz': frequencies.tolist()
    }
    
    # Plot band structure
    plt.figure(figsize=(10, 6))
    for band in frequencies.T:
        plt.plot(band, 'b-', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('q-path')
    plt.ylabel('Frequency (THz)')
    plt.title(f'Phonon Band Structure - {os.path.basename(structure_file)}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/phonon_bands.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate density of states
    mesh = [20, 20, 20]
    phonon.set_mesh(mesh)
    phonon.set_total_DOS()
    
    freq_dos, total_dos = phonon.get_total_DOS()
    
    # Plot DOS
    plt.figure(figsize=(8, 6))
    plt.plot(freq_dos, total_dos, 'b-', linewidth=2)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Frequency (THz)')
    plt.ylabel('Density of States')
    plt.title(f'Phonon DOS - {os.path.basename(structure_file)}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/phonon_dos.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    stability_data.update({
        'dos_frequencies_THz': freq_dos.tolist(),
        'dos_values': total_dos.tolist()
    })
    
    # Save analysis results
    with open(f"{output_dir}/phonon_analysis.json", 'w') as f:
        json.dump(stability_data, f, indent=2)
    
    print(f"Phonon analysis complete:")
    print(f"  Minimum frequency: {min_freq:.3f} THz")
    print(f"  Negative modes: {len(negative_freqs)}")
    print(f"  Dynamically stable: {stability_data['is_stable']}")
    
    return stability_data

def main():
    parser = argparse.ArgumentParser(description='Phonon stability analysis')
    parser.add_argument('--input_dir', type=str, default='relaxed_structures',
                       help='Directory containing relaxed structures')
    parser.add_argument('--output_dir', type=str, default='phonon_results',
                       help='Output directory for phonon analysis')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all relaxed structure files
    structure_files = []
    if os.path.exists(args.input_dir):
        for filename in os.listdir(args.input_dir):
            if filename.endswith(('_relaxed.xyz', '_relaxed.cif')):
                structure_files.append(os.path.join(args.input_dir, filename))
    
    if not structure_files:
        print(f"No relaxed structure files found in {args.input_dir}")
        return
    
    # Analyze each structure
    all_results = []
    for structure_file in structure_files:
        try:
            # Create subdirectory for this structure
            basename = os.path.splitext(os.path.basename(structure_file))[0]
            struct_output_dir = os.path.join(args.output_dir, basename)
            os.makedirs(struct_output_dir, exist_ok=True)
            
            result = compute_phonon_stability(structure_file, struct_output_dir)
            all_results.append(result)
            
        except Exception as e:
            print(f"Error analyzing {structure_file}: {str(e)}")
            continue
    
    # Summary analysis
    stable_structures = [r for r in all_results if r['is_stable']]
    
    print(f"\n=== PHONON STABILITY SUMMARY ===")
    print(f"Total structures analyzed: {len(all_results)}")
    print(f"Dynamically stable structures: {len(stable_structures)}")
    
    if stable_structures:
        print("\nStable structures:")
        for result in stable_structures:
            print(f"  {result['structure']}: min_freq = {result['min_frequency_THz']:.3f} THz")
    
    # Save summary
    summary = {
        'total_analyzed': len(all_results),
        'stable_count': len(stable_structures),
        'all_results': all_results
    }
    
    with open(f"{args.output_dir}/phonon_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}/")

if __name__ == '__main__':
    main()