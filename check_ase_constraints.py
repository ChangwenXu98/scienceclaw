#!/usr/bin/env python3

import ase
from ase import constraints
print(f"ASE version: {ase.__version__}")
print("Available constraints:")
for attr in dir(constraints):
    if not attr.startswith('_'):
        print(f"  {attr}")

# Check specific imports
try:
    from ase.constraints import ExpCellFilter
    print("ExpCellFilter: AVAILABLE")
except ImportError as e:
    print(f"ExpCellFilter: NOT AVAILABLE - {e}")

try:
    from ase.constraints import UnitCellFilter
    print("UnitCellFilter: AVAILABLE")
except ImportError as e:
    print(f"UnitCellFilter: NOT AVAILABLE - {e}")

try:
    from ase.constraints import FixSymmetry
    print("FixSymmetry: AVAILABLE")
except ImportError as e:
    print(f"FixSymmetry: NOT AVAILABLE - {e}")

# Check for alternative imports
try:
    from ase.optimize import ExpCellFilter
    print("ExpCellFilter in ase.optimize: AVAILABLE")
except ImportError as e:
    print(f"ExpCellFilter in ase.optimize: NOT AVAILABLE - {e}")

try:
    from ase.filters import UnitCellFilter
    print("UnitCellFilter in ase.filters: AVAILABLE")
except ImportError as e:
    print(f"UnitCellFilter in ase.filters: NOT AVAILABLE - {e}")

try:
    from ase.filters import ExpCellFilter
    print("ExpCellFilter in ase.filters: AVAILABLE")
except ImportError as e:
    print(f"ExpCellFilter in ase.filters: NOT AVAILABLE - {e}")
