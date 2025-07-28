#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.getcwd())

print("Testing build artifacts validation...")

try:
    from tests.utils.build import BuildTestUtils
    
    print("1. Testing current build artifacts:")
    results = BuildTestUtils.validate_build_artifacts()
    for name, status in results.items():
        print(f"  {name}: {status}")
    
    print("\n2. Testing auto-copy function:")
    success = BuildTestUtils.auto_copy_python_extension()
    print(f"  Auto-copy result: {success}")
    
    print("\n3. Re-testing after auto-copy:")
    results2 = BuildTestUtils.validate_build_artifacts()
    for name, status in results2.items():
        print(f"  {name}: {status}")
        
except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()
