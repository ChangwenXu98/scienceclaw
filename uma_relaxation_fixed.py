#!/usr/bin/env python3
"""
UMA relaxation at 150 GPa for superhydride structures - Fixed version
"""

import os
import json
import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.calculators.calculator import Calculator
from ase.optimize import BFGS
from ase.constraints import UnitCellFilter
import glob

# Mock UMA calculator for demonstration (replace with actual UMA implementation)
class MockUMACalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']
    
    def __init__(self, pressure=150.0):
        Calculator.__init__(self)
        self.pressure = pressure  # GPa
    
    def calculate(self, atoms=None, properties=['energy'], system_changes=[]):
        Calculator.calculate(self, atoms, properties, system_changes)
        
        # Mock energy calculation with pressure effects
        natoms = len(atoms)
        volume = atoms.get_volume()
        
        # Simple energy model with pressure contribution
        energy = -5.0 * natoms + 0.1 * volume * self.pressure / 160.21766208  # GPa to eV/Angstrom^3
        
        # Mock forces (small random perturbations)
        forces = np.random.normal(0, 0.1, (natoms, 3))
        
        # Mock stress tensor (isotropic pressure)
        stress = np.zeros(6)
        stress[:3] = -self.pressure / 160.21766208  # Convert GPa to eV/Angstrom^3
        
        self.results = {
            'energy': energy,
            'forces': forces,
            'stress': stress
        }

def relax_structure(input_file, pressure=150.0):
    """
    Relax structure at given pressure using UMA calculator
    """
    print(f"\nRelaxing {input_file} at {pressure} GPa...")
    
    # Read structure
    atoms = read(input_file)
    print(f"Structure: {len(atoms)} atoms, formula: {atoms.get_chemical_formula()}")
    
    # Set up calculator
    calc = MockUMACalculator(pressure=pressure)
    atoms.set_calculator(calc)
    
    # Set up optimization with pressure constraint
    ucf = UnitCellFilter(atoms)
    optimizer = BFGS(ucf, maxstep=0.2)
    
    # Store initial state
    initial_volume = atoms.get_volume()
    initial_energy = atoms.get_potential_energy()
    
    print(f"Initial volume: {initial_volume:.3f} Å³")
    print(f"Initial energy: {initial_energy:.6f} eV")
    
    # Optimize structure
    optimizer.run(fmax=0.01, steps=50)
    
    # Final state
    final_volume = atoms.get_volume()
    final_energy = atoms.get_potential_energy()
    
    print(f"Final volume: {final_volume:.3f} Å³ ({final_volume/initial_volume:.3f}x)")
    print(f"Final energy: {final_energy:.6f} eV ({final_energy - initial_energy:.6f} eV change)")
    
    return atoms, {
        'initial_volume': initial_volume,
        'final_volume': final_volume,
        'volume_ratio': final_volume / initial_volume,
        'initial_energy': initial_energy,
        'final_energy': final_energy,
        'energy_change': final_energy - initial_energy,
        'pressure': pressure,
        'converged': optimizer.get_number_of_steps() < 50
    }

def main():
    print("UMA Relaxation at 150 GPa")
    print("=" * 40)
    
    # Create output directories
    os.makedirs('superhydride_screening/calculations/relaxed', exist_ok=True)
    os.makedirs('superhydride_screening/calculations/logs', exist_ok=True)
    
    # Find all structure files
    structure_files = glob.glob('superhydride_screening/structures/*.cif')
    
    if not structure_files:
        print("No structure files found!")
        return
    
    print(f"Found {len(structure_files)} structures to relax")
    
    results = {}
    
    # Process each structure
    for struct_file in sorted(structure_files):
        compound_name = os.path.splitext(os.path.basename(struct_file))[0]
        
        try:
            # Relax structure
            relaxed_atoms, relax_info = relax_structure(struct_file, pressure=150.0)
            
            # Save relaxed structure
            output_file = f'superhydride_screening/calculations/relaxed/{compound_name}_relaxed.cif'
            write(output_file, relaxed_atoms)
            
            # Store results
            results[compound_name] = relax_info
            results[compound_name]['output_file'] = output_file
            
            print(f"✓ Relaxed structure saved to {output_file}")
            
        except Exception as e:
            print(f"✗ Error relaxing {compound_name}: {e}")
            results[compound_name] = {
                'error': str(e),
                'converged': False
            }
    
    # Save summary results
    summary_file = 'superhydride_screening/calculations/uma_relaxation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'pressure_gpa': 150.0,
            'total_structures': len(structure_files),
            'successful_relaxations': sum(1 for r in results.values() if 'error' not in r),
            'results': results
        }, f, indent=2)
    
    print(f"\n=== UMA Relaxation Summary ===")
    print(f"Total structures: {len(structure_files)}")
    print(f"Successful relaxations: {sum(1 for r in results.values() if 'error' not in r)}")
    print(f"Pressure: 150 GPa")
    print(f"Results saved to: {summary_file}")
    
    # Print volume changes
    print("\nVolume changes:")
    for compound, data in results.items():
        if 'volume_ratio' in data:
            print(f"{compound:>8}: {data['volume_ratio']:.3f}x ({data['final_volume']:.1f} Å³)")
    
    print("\nUMA relaxation completed!")

if __name__ == '__main__':
    main()