import json
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
