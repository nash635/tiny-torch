#!/usr/bin/env python3
"""
Phase verification script for Tiny-Torch development.

Usage:
    python tools/verify_phase.py 1.1    # Verify Phase 1.1
    python tools/verify_phase.py 1.2    # Verify Phase 1.2
    python tools/verify_phase.py all    # Verify all implemented phases
"""

import sys
import subprocess
from pathlib import Path

def run_phase_verification(phase):
    """Run verification for a specific phase."""
    project_root = Path(__file__).parent.parent
    
    if phase == "all":
        # Run all phase verification tests
        cmd = [
            sys.executable, "-m", "pytest", 
            "tests/system/test_complete.py::TestPhaseVerification",
            "-v", "-s"
        ]
    else:
        # Run specific phase verification
        phase_underscored = phase.replace(".", "_")
        test_name = f"test_phase_{phase_underscored}_verification"
        cmd = [
            sys.executable, "-m", "pytest", 
            f"tests/system/test_complete.py::TestPhaseVerification::{test_name}",
            "-v", "-s"
        ]
    
    print(f"Running Phase {phase} verification...")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, cwd=project_root, check=False)
        return result.returncode
    except Exception as e:
        print(f"Error running verification: {e}")
        return 1

def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    
    phase = sys.argv[1]
    
    if phase not in ["1.1", "1.2", "1.3", "2.1", "2.2", "3.1", "all"]:
        print(f"Unknown phase: {phase}")
        print("Available phases: 1.1, 1.2, 1.3, 2.1, 2.2, 3.1, all")
        return 1
    
    return run_phase_verification(phase)

if __name__ == "__main__":
    sys.exit(main())
