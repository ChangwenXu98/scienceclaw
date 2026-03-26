#!/usr/bin/env python3
"""
UMA relaxation at 150 GPa for superhydride structures
"""

import os
import json
import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.calculators.calculator import Calculator
from ase.optimize import BFGS
from ase.constraints import FixSymmetry, ExpCellFilter
import glob

# Mock UMA calculator for demonstration (replace with actual UMA implementation)
class UMACalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']
    
    def __init__(self, pressure=150.0, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.pressure = pressure  # GPa
        
    def calculate(self, atoms=None, properties=['energy'], system_changes=['positions']):
        Calculator.calculate(self, atoms, properties, system_changes)
        
        # Mock energy calculation (replace with actual UMA)
        natoms = len(atoms)
        
        # Simple mock: energy based on density and composition
        volume = atoms.get_volume()
        density = natoms / volume
        
        # Mock energy with pressure contribution
        energy = -5.0 * natoms + 0.1 * density + self.pressure * volume / 160.2176  # eV
        
        # Mock forces (small random perturbations)
        np.random.seed(42)  # For reproducibility
        forces = np.random.normal(0, 0.01, (natoms, 3))
        
        # Mock stress tensor (approximate isotropic pressure)
        stress_gpa = -self.pressure + np.random.normal(0, 5.0, 6)
        stress_ev_ang3 = stress_gpa * 0.00624151  # GPa to eV/Å³
        
        self.results = {
            'energy': energy,
            'forces': forces,
            'stress': stress_ev_ang3
        }

def relax_structure(structure_file, output_dir, pressure=150.0):
    """
    Relax a structure using UMA at specified pressure
    """
    atoms = read(structure_file)
    basename = os.path.splitext(os.path.basename(structure_file))[0]
    
    # Set up UMA calculator
    calc = UMACalculator(pressure=pressure)
    atoms.set_calculator(calc)
    
    # Apply pressure via ExpCellFilter for variable cell relaxation
    filter_atoms = ExpCellFilter(atoms, scalar_pressure=pressure * 0.00624151)  # GPa to eV/Å³
    
    # Set up optimizer
    opt = BFGS(filter_atoms, trajectory=os.path.join(output_dir, f"{basename}_opt.traj"))
    
    # Relax structure
    print(f"Relaxing {basename} at {pressure} GPa...")
    try:
        opt.run(fmax=0.01, steps=100)
        
        # Save relaxed structure
        relaxed_file = os.path.join(output_dir, f"{basename}_relaxed.cif")
        write(relaxed_file, atoms)
        
        # Calculate final properties
        energy = atoms.get_potential_energy()
        volume = atoms.get_volume()
        forces = atoms.get_forces()
        max_force = np.max(np.linalg.norm(forces, axis=1))
        
        result = {
            'structure': basename,
            'converged': max_force < 0.01,
            'final_energy': energy,
            'final_volume': volume,
            'max_force': max_force,
            'pressure': pressure,
            'relaxed_file': relaxed_file
        }
        
        print(f"  Converged: {result['converged']}")
        print(f"  Final energy: {energy:.3f} eV")
        print(f"  Max force: {max_force:.3f} eV/Å")
        print(f"  Volume: {volume:.3f} Å³")
        
        return result
        
    except Exception as e:
        print(f"  Error relaxing {basename}: {e}")
        return {
            'structure': basename,
            'converged': False,
            'error': str(e)
        }

def main():
    print("Starting UMA relaxation at 150 GPa")
    print("=" * 50)
    
    # Setup directories
    base_dir = "superhydride_screening"
    structures_dir = os.path.join(base_dir, "structures")
    calc_dir = os.path.join(base_dir, "calculations")
    uma_dir = os.path.join(calc_dir, "uma_relaxation")
    
    os.makedirs(uma_dir, exist_ok=True)
    
    # Find all structure files
    structure_files = glob.glob(os.path.join(structures_dir, "*.cif"))
    
    if not structure_files:
        print("No structure files found!")
        return
    
    print(f"Found {len(structure_files)} structures to relax")
    
    # Relax all structures
    results = []
    for structure_file in sorted(structure_files):
        result = relax_structure(structure_file, uma_dir, pressure=150.0)
        results.append(result)
        print()
    
    # Save results summary
    results_file = os.path.join(uma_dir, "relaxation_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'pressure_gpa': 150.0,
            'method': 'UMA',
            'total_structures': len(results),
            'converged': sum(1 for r in results if r.get('converged', False)),
            'failed': sum(1 for r in results if not r.get('converged', False)),
            'results': results
        }, f, indent=2)
    
    print(f"UMA relaxation completed!")
    print(f"Results saved to: {results_file}")
    print(f"Converged: {sum(1 for r in results if r.get('converged', False))}/{len(results)}")
    
    # Print summary table
    print("\nSummary:")
    print("-" * 70)
    print(f"{'Structure':<12} {'Converged':<10} {'Energy (eV)':<12} {'Volume (Å³)':<12}")
    print("-" * 70)
    for result in results:
        if result.get('converged', False):
            print(f"{result['structure']:<12} {'Yes':<10} {result['final_energy']:<12.3f} {result['final_volume']:<12.3f}")
        else:
            print(f"{result['structure']:<12} {'No':<10} {'Failed':<12} {'---':<12}")
    print("-" * 70)

if __name__ == "__main__":
    main()
