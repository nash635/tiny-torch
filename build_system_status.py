#!/usr/bin/env python3
"""
Tiny-Torch Quick Status Check

Lightweight status check for development workflow.
For detailed diagnostics, use: make diagnose
"""

import sys
import subprocess
from pathlib import Path

def quick_check():
    print("🔧 Tiny-Torch Quick Status")
    print("=" * 30)
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Not in tiny-torch root directory")
        return False
    
    # Check Python version
    if sys.version_info < (3, 8):
        print(f"❌ Python {sys.version_info[:2]} < 3.8")
        return False
    print(f"✅ Python {sys.version_info[:2]}")
    
    # Check key tools
    tools = {"cmake": "CMake", "ninja": "Ninja", "make": "Make"}
    missing = []
    for cmd, name in tools.items():
        try:
            subprocess.run([cmd, "--version"], capture_output=True, check=True, timeout=3)
            print(f"✅ {name}")
        except:
            print(f"❌ {name}")
            missing.append(name)
    
    # Check build status
    if Path("build").exists():
        print("📁 Build directory exists")
    else:
        print("📂 No build directory")
    
    # Check if installed
    try:
        import tiny_torch
        print("✅ tiny_torch importable")
    except ImportError:
        print("❌ tiny_torch not installed")
    
    # Summary
    print("\n" + "=" * 30)
    if missing:
        print(f"⚠️  Missing tools: {', '.join(missing)}")
        print("💡 Run 'make diagnose' for detailed help")
    else:
        print("🎉 All essential tools available!")
        print("💡 Ready for 'make build' or 'make test'")
    
    return len(missing) == 0

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        print("\nUsage: python build_system_status.py")
        print("For detailed diagnostics: make diagnose")
        sys.exit(0)
    
    success = quick_check()
    sys.exit(0 if success else 1)
