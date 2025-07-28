#!/usr/bin/env python3
"""
CUDA environment detection script for tests.
Returns exit code 0 if CUDA is available, 1 otherwise.
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    """Check CUDA availability using existing test utils."""
    try:
        from tests.utils.cuda import CudaTestUtils
        
        # Check if explicitly disabled
        env_cuda = os.getenv("WITH_CUDA", "").lower()
        if env_cuda in ["0", "false", "off", "no"]:
            print("CUDA disabled by environment variable")
            return False
        
        # Use existing CUDA detection logic
        info = CudaTestUtils.get_cuda_info()
        
        if info['driver_available'] and info['nvcc_available']:
            print("CUDA available: nvidia-smi and nvcc found")
            return True
        elif info['driver_available']:
            print("CUDA available: nvidia-smi found (nvcc not available)")
            return True
        else:
            print("CUDA not available")
            return False
            
    except Exception as e:
        print(f"Error checking CUDA: {e}")
        return False


if __name__ == "__main__":
    if main():
        sys.exit(0)  # CUDA available
    else:
        sys.exit(1)  # CUDA not available
