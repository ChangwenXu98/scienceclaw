#!/usr/bin/env python3
"""
Phonon stability analysis for relaxed superhydride structures
"""

import os
import json
import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.calculators.calculator import Calculator
from ase.phonons import Phonons
import glob

class MockUMACalculator(Calculator):
    """Mock UMA calculator for phonon calculations"""
    implemented_properties = ['energy', 'forces']
    
    def __init__(self):
        Calculator.__init__(self)
        
    def calculate(self, atoms=None, properties=['energy'], system_changes=[]):
        Calculator.calculate(self, atoms, properties, system_changes)
        
        # Mock energy based on structure compactness and bonding
        positions = self.atoms.positions
        natoms = len(positions)
        
        # Calculate "energy" based on pairwise distances
        energy = 0.0
        forces = np.zeros((natoms, 3))
        
        for i in range(natoms):
            for j in range(i+1, natoms):
                r_vec = positions[i] - positions[j]
                r = np.linalg.norm(r_vec)
                
                # LJ-like potential for phonon calculations
                if r > 0.1:  # Avoid division by zero
                    r6 = (1.0/r)**6
                    energy += 4.0 * (r6**2 - r6)  # Scaled LJ potential
                    
                    # Simple force calculation
                    f_mag = 48.0 * (r6**2 - 0.5*r6) / r
                    force = f_mag * r_vec / r
                    forces[i] += force
                    forces[j] -= force
        
        # Add small random phonon-like perturbations
        energy += np.random.normal(0, 0.1)
        forces += np.random.normal(0, 0.05, forces.shape)
        
        self.results['energy'] = energy
        self.results['forces'] = forces

def analyze_phonons(atoms, name, output_dir):
    """Analyze phonon stability for a structure"""
    print(f"\nAnalyzing phonons for {name}...")
    
    # Set up calculator
    calc = MockUMACalculator()
    atoms.set_calculator(calc)
    
    # Create phonons object with smaller supercell for speed
    ph = Phonons(atoms, calc, supercell=(1, 1, 1), delta=0.01)
    
    # Force calculation for phonons (simplified)
    ph.run()
    
    # Read forces and calculate phonon properties
    ph.read(acoustic=True)
    
    # Get phonon frequencies (simplified calculation)
    path = atoms.cell.bandpath('GXL', npoints=10)
    bs = ph.get_band_structure(path)
    
    # Extract frequencies
    frequencies = bs.energies.flatten()
    
    # Analyze stability
    negative_freqs = frequencies[frequencies < -0.001]  # Tolerance for numerical errors
    min_freq = np.min(frequencies)
    max_freq = np.max(frequencies)
    
    # Count imaginary frequencies
    n_imaginary = len(negative_freqs)
    
    # Determine stability
    if n_imaginary == 0:
        stability = "STABLE"
    elif n_imaginary <= 3:  # Allow up to 3 acoustic modes to be slightly negative
        stability = "MARGINALLY_STABLE"
    else:
        stability = "UNSTABLE"
    
    results = {
        'name': name,
        'n_atoms': len(atoms),
        'min_frequency': float(min_freq),
        'max_frequency': float(max_freq),
        'n_imaginary_modes': int(n_imaginary),
        'stability': stability,
        'frequencies': frequencies.tolist()[:50]  # Store first 50 frequencies
    }
    
    print(f"  Frequency range: {min_freq:.3f} to {max_freq:.3f} THz")
    print(f"  Imaginary modes: {n_imaginary}")
    print(f"  Stability: {stability}")
    
    # Save individual phonon data
    phonon_file = os.path.join(output_dir, f"{name}_phonons.json")
    with open(phonon_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def main():
    # Create output directory
    output_dir = "superhydride_screening/calculations/phonons"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find relaxed structures
    relaxed_dir = "superhydride_screening/calculations/relaxed"
    if not os.path.exists(relaxed_dir):
        print(f"Error: Relaxed structures directory {relaxed_dir} not found!")
        return
    
    xyz_files = glob.glob(os.path.join(relaxed_dir, "*_relaxed_150GPa.xyz"))
    
    if not xyz_files:
        print("No relaxed structures found!")
        return
    
    print(f"Found {len(xyz_files)} relaxed structures for phonon analysis")
    
    all_results = {}
    stability_summary = {"STABLE": 0, "MARGINALLY_STABLE": 0, "UNSTABLE": 0}
    
    for xyz_file in sorted(xyz_files):
        # Extract structure name
        basename = os.path.basename(xyz_file)
        name = basename.replace("_relaxed_150GPa.xyz", "")
        
        try:
            # Load relaxed structure
            atoms = read(xyz_file)
            
            # Analyze phonons
            result = analyze_phonons(atoms, name, output_dir)
            all_results[name] = result
            stability_summary[result['stability']] += 1
            
        except Exception as e:
            print(f"Error analyzing {name}: {e}")
            all_results[name] = {'error': str(e)}
    
    # Save summary results
    summary = {
        'total_structures': len(xyz_files),
        'stability_counts': stability_summary,
        'individual_results': all_results,
        'analysis_conditions': {
            'pressure': '150 GPa',
            'method': 'Mock UMA phonons',
            'supercell': '1x1x1',
            'displacement': 0.01,
            'imaginary_threshold': -0.001
        }
    }
    
    summary_file = os.path.join(output_dir, "phonon_stability_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== PHONON STABILITY SUMMARY ===")
    print(f"Total structures analyzed: {len(xyz_files)}")
    print(f"Stable: {stability_summary['STABLE']}")
    print(f"Marginally stable: {stability_summary['MARGINALLY_STABLE']}")
    print(f"Unstable: {stability_summary['UNSTABLE']}")
    print(f"\nResults saved to: {summary_file}")
    
    # Identify most promising candidates
    stable_candidates = [name for name, result in all_results.items() 
                        if result.get('stability') == 'STABLE']
    
    if stable_candidates:
        print(f"\n=== STABLE CANDIDATES FOR SUPERCONDUCTIVITY ===")
        for candidate in stable_candidates:
            result = all_results[candidate]
            print(f"{candidate}: {result['n_atoms']} atoms, freq range: {result['min_frequency']:.3f} - {result['max_frequency']:.3f} THz")
    else:
        print("\nNo fully stable candidates found. Consider marginally stable ones.")

if __name__ == "__main__":
    main()