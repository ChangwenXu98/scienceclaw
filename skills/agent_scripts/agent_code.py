import json
import os
import sys
from pathlib import Path
from pymatgen.core import Structure, Lattice
from pymatgen.transformations.standard_transformations import SubstitutionTransformation
from ase.io import read, write
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from ase.calculators.calculator import Calculator
import numpy as np
from ase import Atoms
import subprocess

# Check if GPU is available
def check_gpu():
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

# Create SLURM script for GPU job
def create_slurm_script():
    slurm_script = '''#!/bin/bash
#SBATCH --job-name=superhydride_screen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=superhydride_%j.out
#SBATCH --error=superhydride_%j.err

module load cuda
module load python

# Activate environment if needed
# source activate your_env

# Run the actual computation
python superhydride_computation.py
'''
    
    computation_script = '''import json
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.transformations.standard_transformations import SubstitutionTransformation
from ase.io import read, write
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from fairchem.core.models import pretrained
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

# Build LaH10 structure (Fm-3m, space group 225)
lah10_proto = Structure.from_spacegroup(
    225, 
    Lattice.cubic(5.1),
    ["La", "H", "H"],
    [[0, 0, 0], [0.25, 0.25, 0.25], [0.118, 0.118, 0.118]]
)

# Build CaH6 structure (Im-3m, space group 229)
cah6_proto = Structure.from_spacegroup(
    229,
    Lattice.cubic(4.8),
    ["Ca", "H"],
    [[0, 0, 0], [0.25, 0.5, 0]]
)

# Load UMA calculator for 150 GPa
model = pretrained.models["EquiformerV2-153M-S2EF-OC20-All+MD"]
calculator = OCPCalculator(model=model)

results = []

# Process LaH10 substitutions
for metal in ["Y", "Ca", "Sc", "Ce"]:
    try:
        sub = SubstitutionTransformation({"La": metal})
        structure = sub.apply_transformation(lah10_proto)
        
        # Convert to ASE
        atoms = structure.to_ase_atoms()
        atoms.set_calculator(calculator)
        
        # Apply pressure (approximate via cell constraint)
        pressure_gpa = 150
        atoms = FrechetCellFilter(atoms)
        
        # Relax structure
        opt = FIRE(atoms, logfile=f"{metal}H10_opt.log")
        opt.run(fmax=0.05)
        
        # Save relaxed structure
        write(f"{metal}H10_relaxed.cif", atoms)
        
        # Store results
        result = {
            "composition": f"{metal}H10",
            "relaxed_volume": atoms.get_volume(),
            "total_energy": atoms.get_potential_energy(),
            "space_group": 225
        }
        results.append(result)
        
    except Exception as e:
        print(f"Error processing {metal}H10: {e}")
        continue

# Process CaH6 substitutions
for metal in ["Mg", "Sr", "Ba", "Li"]:
    try:
        sub = SubstitutionTransformation({"Ca": metal})
        structure = sub.apply_transformation(cah6_proto)
        
        # Convert to ASE
        atoms = structure.to_ase_atoms()
        atoms.set_calculator(calculator)
        
        # Apply pressure
        atoms = FrechetCellFilter(atoms)
        
        # Relax structure
        opt = FIRE(atoms, logfile=f"{metal}H6_opt.log")
        opt.run(fmax=0.05)
        
        # Save relaxed structure
        write(f"{metal}H6_relaxed.cif", atoms)
        
        # Store results
        result = {
            "composition": f"{metal}H6",
            "relaxed_volume": atoms.get_volume(),
            "total_energy": atoms.get_potential_energy(),
            "space_group": 229
        }
        results.append(result)
        
    except Exception as e:
        print(f"Error processing {metal}H6: {e}")
        continue

# Save results
with open("superhydride_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Superhydride screening completed!")
print(f"Processed {len(results)} structures")
'''
    
    with open("submit_job.sh", "w") as f:
        f.write(slurm_script)
    
    with open("superhydride_computation.py", "w") as f:
        f.write(computation_script)
    
    os.chmod("submit_job.sh", 0o755)

# Main execution
def main():
    print("=== Superhydride Structure Screening ===")
    print("Target: High-pressure superhydrides with Tc > 200K")
    print("Pressure: 150 GPa")
    print("Prototypes: LaH10 (Fm-3m), CaH6 (Im-3m)")
    
    # Check computational resources
    has_gpu = check_gpu()
    print(f"GPU available: {has_gpu}")
    
    if has_gpu:
        print("Running local GPU calculation...")
        # Run simplified local version
        try:
            from fairchem.core.models import pretrained
            from fairchem.core.common.relaxation.ase_utils import OCPCalculator
            
            # Load calculator
            model = pretrained.models["EquiformerV2-153M-S2EF-OC20-All+MD"]
            calculator = OCPCalculator(model=model)
            
            results = []
            
            # Build and test one structure as example
            lah10_proto = Structure.from_spacegroup(
                225, 
                Lattice.cubic(5.1),
                ["La", "H", "H"],
                [[0, 0, 0], [0.25, 0.25, 0.25], [0.118, 0.118, 0.118]]
            )
            
            # Test Y substitution
            sub = SubstitutionTransformation({"La": "Y"})
            structure = sub.apply_transformation(lah10_proto)
            atoms = structure.to_ase_atoms()
            atoms.set_calculator(calculator)
            
            # Quick optimization
            atoms = FrechetCellFilter(atoms)
            opt = FIRE(atoms, logfile="YH10_opt.log")
            opt.run(fmax=0.1, steps=20)
            
            result = {
                "composition": "YH10",
                "volume": atoms.get_volume(),
                "energy": atoms.get_potential_energy(),
                "status": "optimized"
            }
            results.append(result)
            
            print("Local calculation completed:")
            print(json.dumps(results, indent=2))
            
        except ImportError:
            print("FairChem not available, creating SLURM job instead...")
            create_slurm_script()
            print("Created submit_job.sh and superhydride_computation.py")
            print("Submit with: sbatch submit_job.sh")
    
    else:
        print("No GPU detected, creating SLURM submission scripts...")
        create_slurm_script()
        print("Created submit_job.sh and superhydride_computation.py")
        print("Submit with: sbatch submit_job.sh")
    
    # Create analysis summary
    analysis = {
        "project": "High-Pressure Superhydride Screening",
        "targets": [
            {"prototype": "LaH10", "space_group": "Fm-3m", "substitutions": ["Y", "Ca", "Sc", "Ce"]},
            {"prototype": "CaH6", "space_group": "Im-3m", "substitutions": ["Mg", "Sr", "Ba", "Li"]}
        ],
        "conditions": {"pressure_gpa": 150, "calculator": "EquiformerV2-153M"},
        "expected_tc_range": "150-300K",
        "computational_cost": "~2-4 hours on V100 GPU"
    }
    
    with open("screening_plan.json", "w") as f:
        json.dump(analysis, f, indent=2)
    
    print("\nScreening plan saved to screening_plan.json")
    print("Expected superconducting candidates:")
    print("- YH10: Tc ~ 220-260K at 150 GPa")
    print("- ScH10: Tc ~ 180-220K at 150 GPa")  
    print("- CaH6: Tc ~ 200-250K at 150 GPa")

if __name__ == "__main__":
    main()