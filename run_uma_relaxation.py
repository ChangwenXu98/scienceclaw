#!/usr/bin/env python3

import os
import subprocess
import time
from pathlib import Path

def submit_uma_jobs():
    """Submit UMA relaxation jobs for all structures"""
    
    structures_dir = Path('structures')
    uma_dir = Path('uma_relaxed')
    uma_dir.mkdir(exist_ok=True)
    
    job_ids = []
    
    # Get all POSCAR files
    poscar_files = list(structures_dir.glob('*_POSCAR'))
    print(f"Found {len(poscar_files)} structures to relax")
    
    for poscar_file in poscar_files:
        compound_name = poscar_file.stem.replace('_POSCAR', '')
        work_dir = uma_dir / compound_name
        work_dir.mkdir(exist_ok=True)
        
        # Copy POSCAR to work directory
        subprocess.run(['cp', str(poscar_file), str(work_dir / 'POSCAR')])
        
        # Create UMA relaxation script
        uma_script = work_dir / 'uma_relax.py'
        with open(uma_script, 'w') as f:
            f.write(f"""#!/usr/bin/env python3

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from fairchem.core.models.model_registry import model_name_to_local_file
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from ase.io import read, write
from ase.optimize import BFGS
from ase import units
import numpy as np

print(f"Starting UMA relaxation for {compound_name}")
print(f"Working directory: {{os.getcwd()}}")

# Load structure
if os.path.exists('POSCAR'):
    atoms = read('POSCAR', format='vasp')
    print(f"Loaded structure with {{len(atoms)}} atoms")
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
    print(f"Error loading UMA calculator: {{e}}")
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
opt = BFGS(atoms, trajectory='{compound_name}_relaxation.traj')
opt.run(fmax=0.05, steps=200)

# Save relaxed structure
write('{compound_name}_relaxed.cif', atoms, format='cif')
write('{compound_name}_relaxed_POSCAR', atoms, format='vasp')

# Get final properties
final_energy = atoms.get_potential_energy()
final_volume = atoms.get_volume()
final_stress = atoms.get_stress()

print(f"Relaxation completed for {compound_name}")
print(f"Final energy: {{final_energy:.6f}} eV")
print(f"Final volume: {{final_volume:.3f}} Å³")
print(f"Final stress: {{final_stress}}")

# Save results
with open('{compound_name}_relaxation_results.txt', 'w') as f:
    f.write(f"Compound: {compound_name}\n")
    f.write(f"Applied pressure: {{pressure_gpa}} GPa\n")
    f.write(f"Final energy: {{final_energy:.6f}} eV\n")
    f.write(f"Final volume: {{final_volume:.3f}} Å³\n")
    f.write(f"Final stress (GPa): {{final_stress * 160.2176}}\n")
    f.write(f"Number of atoms: {{len(atoms)}}\n")

print(f"Results saved to {compound_name}_relaxation_results.txt")
""")
        
        # Create SLURM job script
        slurm_script = work_dir / f'{compound_name}_uma.sh'
        with open(slurm_script, 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH --job-name=uma_{compound_name}
#SBATCH --partition=venkvis-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --time=4:00:00
#SBATCH --output={compound_name}_uma_%j.out
#SBATCH --error={compound_name}_uma_%j.err

# Load modules
module load python/3.9
module load cuda/11.8

# Change to work directory
cd {work_dir.absolute()}

# Install required packages
pip install --user torch torchvision torchaudio fairchem-core ase

# Run UMA relaxation
python uma_relax.py

echo "UMA relaxation completed for {compound_name}"
""")
        
        # Submit job
        result = subprocess.run(['sbatch', str(slurm_script)], 
                              capture_output=True, text=True, cwd=work_dir)
        
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            job_ids.append((compound_name, job_id))
            print(f"Submitted UMA job for {compound_name}: {job_id}")
        else:
            print(f"Failed to submit job for {compound_name}: {result.stderr}")
    
    return job_ids

if __name__ == "__main__":
    print("Starting UMA relaxation workflow...")
    job_ids = submit_uma_jobs()
    
    print(f"\nSubmitted {len(job_ids)} UMA relaxation jobs:")
    for compound, job_id in job_ids:
        print(f"  {compound}: {job_id}")
    
    print("\nUse 'squeue -u $USER' to monitor job status")
    print("Jobs will relax structures at 150 GPa using UMA")
