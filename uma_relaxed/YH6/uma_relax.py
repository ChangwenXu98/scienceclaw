#!/usr/bin/env python3

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from fairchem.core.models.model_registry import model_name_to_local_file
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from ase.io import read, write
from ase.optimize import BFGS
from ase import units
import numpy as np

print(f"Starting UMA relaxation for YH6")
print(f"Working directory: {os.getcwd()}")

# Load structure
if os.path.exists('POSCAR'):
    atoms = read('POSCAR', format='vasp')
    print(f"Loaded structure with {len(atoms)} atoms")
else:
    print("ERROR: POSCAR not found")
    exit(1)

# Apply 150 GPa pressure (convert to eV/Å³)
pressure_gpa = 150.0
pressure_ev_ang3 = pressure_gpa * 0.00624151  # GPa to eV/Å³

# Set up UMA calculator
try:
    checkpoint_path = model_name_to_local_file('EquiformerV2-31M-S2EF-OC20-All+MD', local_cache='/tmp/ocp_checkpoints/')
    calc = OCPCalculator(checkpoint_path=checkpoint_path)
    atoms.calc = calc
    print("UMA calculator loaded successfully")
except Exception as e:
    print(f"Error loading UMA calculator: {e}")
    exit(1)

# Apply external pressure by modifying stress
class PressureCalculator:
    def __init__(self, calc, pressure):
        self.calc = calc
        self.pressure = pressure
    
    def get_potential_energy(self, atoms):
        energy = self.calc.get_potential_energy(atoms)
        volume = atoms.get_volume()
        # Add PV term for pressure
        energy += self.pressure * volume
        return energy
    
    def get_forces(self, atoms):
        return self.calc.get_forces(atoms)
    
    def get_stress(self, atoms):
        stress = self.calc.get_stress(atoms)
        # Add isotropic pressure
        pressure_stress = np.array([self.pressure, self.pressure, self.pressure, 0, 0, 0])
        return stress + pressure_stress

# Use pressure calculator
atoms.calc = PressureCalculator(calc, pressure_ev_ang3)

# Relax structure
print("Starting structure relaxation...")
opt = BFGS(atoms, trajectory='YH6_relaxation.traj')
opt.run(fmax=0.05, steps=200)

# Save relaxed structure
write('YH6_relaxed.cif', atoms, format='cif')
write('YH6_relaxed_POSCAR', atoms, format='vasp')

# Get final properties
final_energy = atoms.get_potential_energy()
final_volume = atoms.get_volume()
final_stress = atoms.get_stress()

print(f"Relaxation completed for YH6")
print(f"Final energy: {final_energy:.6f} eV")
print(f"Final volume: {final_volume:.3f} Å³")
print(f"Final stress: {final_stress}")

# Save results
with open('YH6_relaxation_results.txt', 'w') as f:
    f.write(f"Compound: YH6
")
    f.write(f"Applied pressure: {pressure_gpa} GPa
")
    f.write(f"Final energy: {final_energy:.6f} eV
")
    f.write(f"Final volume: {final_volume:.3f} Å³
")
    f.write(f"Final stress (GPa): {final_stress * 160.2176}
")
    f.write(f"Number of atoms: {len(atoms)}
")

print(f"Results saved to YH6_relaxation_results.txt")
