#!/usr/bin/env python3
"""
简化的内存工具测试脚本
验证合并后工具的基本功能
"""

import sys
import os

# 添加当前目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_imports():
    """测试模块导入"""
    print("[INFO] 测试模块导入...")
    
    try:
        from memory_debug import MemoryDebugger
        print("   [PASS] MemoryDebugger 导入成功")
    except Exception as e:
        print(f"   [FAIL] MemoryDebugger 导入失败: {e}")
        return False
    
    try:
        from memory_debug import MemoryProfiler, OOMDetector
        print("   [PASS] 核心组件导入成功")
    except Exception as e:
        print(f"   [FAIL] 核心组件导入失败: {e}")
        return False
    
    return True

def test_basic_functionality():
    """测试基本功能"""
    print("\n[INFO] 测试基本功能...")
    
    try:
        from memory_debug import MemoryDebugger
        
        # 创建调试器实例
        debugger = MemoryDebugger()
        print("   [PASS] MemoryDebugger 实例创建成功")
        
        # 测试状态报告
        report = debugger.get_status_report()
        print(f"   [PASS] 状态报告生成成功: {len(report)} 个字段")
        
        return True
    except Exception as e:
        print(f"   [FAIL] 基本功能测试失败: {e}")
        return False

def test_cli_help():
    """测试命令行帮助"""
    print("\n[INFO] 测试命令行接口...")
    
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, 'memory_debug.py', '--help'
        ], cwd=current_dir, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("   [PASS] 命令行帮助显示成功")
            return True
        else:
            print(f"   [FAIL] 命令行测试失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"   [FAIL] 命令行测试异常: {e}")
        return False

def main():
    """主测试函数"""
    print("[INFO] 开始测试合并后的内存调试工具...")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_basic_functionality,
        test_cli_help
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"   [FAIL] 测试异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"[INFO] 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("[PASS] 所有测试通过！内存调试工具整合成功")
        return True
    else:
        print("[WARNING] 部分测试失败，请检查相关问题")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
