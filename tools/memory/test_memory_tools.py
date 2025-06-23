"""
Memory Tools Test Script - 显存调试工具测试脚本
用于验证各个工具模块的基本功能
"""

import os
import sys
import time
import subprocess
import tempfile

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试模块导入"""
    print("🔍 Testing module imports...")
    
    try:
        from memory_profiler import MemoryProfiler
        print("   ✅ MemoryProfiler imported successfully")
    except Exception as e:
        print(f"   ❌ MemoryProfiler import failed: {e}")
        return False
    
    try:
        from oom_detector import OOMDetector
        print("   ✅ OOMDetector imported successfully")
    except Exception as e:
        print(f"   ❌ OOMDetector import failed: {e}")
        return False
    
    try:
        from fragmentation_analyzer import FragmentationAnalyzer
        print("   ✅ FragmentationAnalyzer imported successfully")
    except Exception as e:
        print(f"   ❌ FragmentationAnalyzer import failed: {e}")
        return False
    
    try:
        from memory_leak_detector import MemoryLeakDetector
        print("   ✅ MemoryLeakDetector imported successfully")
    except Exception as e:
        print(f"   ❌ MemoryLeakDetector import failed: {e}")
        return False
    
    try:
        from distributed_memory_monitor import DistributedMemoryMonitor
        print("   ✅ DistributedMemoryMonitor imported successfully")
    except Exception as e:
        print(f"   ❌ DistributedMemoryMonitor import failed: {e}")
        return False
    
    print("   🎉 All modules imported successfully!")
    return True

def test_memory_profiler():
    """测试内存分析器"""
    print("\n🔍 Testing Memory Profiler...")
    
    try:
        from memory_profiler import MemoryProfiler
        
        profiler = MemoryProfiler(sampling_interval=0.5)
        print("   ✅ MemoryProfiler created")
        
        profiler.start_monitoring()
        print("   ✅ Monitoring started")
        
        time.sleep(3)  # 监控3秒
        
        profiler.stop_monitoring()
        print("   ✅ Monitoring stopped")
        
        # 测试获取状态
        status = profiler.get_current_status()
        print(f"   📊 Current status: {len(status)} devices monitored")
        
        # 测试生成报告
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            report = profiler.generate_report(f.name)
            print(f"   📄 Report generated: {f.name}")
            os.unlink(f.name)  # 清理临时文件
        
        return True
        
    except Exception as e:
        print(f"   ❌ Memory Profiler test failed: {e}")
        return False

def test_oom_detector():
    """测试OOM检测器"""
    print("\n🔍 Testing OOM Detector...")
    
    try:
        from oom_detector import OOMDetector, default_warning_callback
        
        detector = OOMDetector(
            warning_threshold=0.8,
            critical_threshold=0.9,
            sampling_interval=1.0
        )
        print("   ✅ OOMDetector created")
        
        # 添加回调
        detector.add_warning_callback(default_warning_callback)
        print("   ✅ Warning callback added")
        
        detector.start_monitoring()
        print("   ✅ Monitoring started")
        
        time.sleep(3)  # 监控3秒
        
        detector.stop_monitoring()
        print("   ✅ Monitoring stopped")
        
        # 测试风险评估
        risks = detector.get_risk_assessment()
        print(f"   ⚠️  Risk assessment: {len(risks)} devices analyzed")
        
        # 测试生成报告
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            report = detector.generate_oom_report(f.name)
            print(f"   📄 Report generated: {f.name}")
            os.unlink(f.name)
        
        return True
        
    except Exception as e:
        print(f"   ❌ OOM Detector test failed: {e}")
        return False

def test_fragmentation_analyzer():
    """测试碎片分析器"""
    print("\n🔍 Testing Fragmentation Analyzer...")
    
    try:
        from fragmentation_analyzer import FragmentationAnalyzer
        
        analyzer = FragmentationAnalyzer(sampling_interval=1.0)
        print("   ✅ FragmentationAnalyzer created")
        
        analyzer.start_monitoring()
        print("   ✅ Monitoring started")
        
        time.sleep(3)  # 监控3秒
        
        analyzer.stop_monitoring()
        print("   ✅ Monitoring stopped")
        
        # 测试获取碎片化状态
        current_frag = analyzer.get_current_fragmentation()
        print(f"   🧩 Fragmentation status: {len(current_frag)} devices analyzed")
        
        # 测试生成报告
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            report = analyzer.generate_fragmentation_report(f.name)
            print(f"   📄 Report generated: {f.name}")
            os.unlink(f.name)
        
        return True
        
    except Exception as e:
        print(f"   ❌ Fragmentation Analyzer test failed: {e}")
        return False

def test_memory_leak_detector():
    """测试内存泄漏检测器"""
    print("\n🔍 Testing Memory Leak Detector...")
    
    try:
        from memory_leak_detector import MemoryLeakDetector
        
        detector = MemoryLeakDetector(
            sampling_interval=1.0,
            leak_threshold=5*1024*1024,  # 5MB/min
            enable_reference_tracking=True
        )
        print("   ✅ MemoryLeakDetector created")
        
        detector.start_monitoring()
        print("   ✅ Monitoring started")
        
        time.sleep(3)  # 监控3秒
        
        detector.stop_monitoring()
        print("   ✅ Monitoring stopped")
        
        # 测试获取泄漏摘要
        summary = detector.get_leak_summary()
        if 'error' not in summary:
            print(f"   🔍 Leak summary: {summary['total_leaks_detected']} leaks detected")
        else:
            print(f"   🔍 Leak summary: {summary['error']}")
        
        # 测试生成报告
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            report = detector.generate_leak_report(f.name)
            print(f"   📄 Report generated: {f.name}")
            os.unlink(f.name)
        
        return True
        
    except Exception as e:
        print(f"   ❌ Memory Leak Detector test failed: {e}")
        return False

def test_distributed_monitor():
    """测试分布式监控器"""
    print("\n🔍 Testing Distributed Memory Monitor...")
    
    try:
        from distributed_memory_monitor import DistributedMemoryMonitor, DistributedRole
        
        # 测试standalone模式
        monitor = DistributedMemoryMonitor(
            role=DistributedRole.STANDALONE,
            sampling_interval=1.0
        )
        print("   ✅ DistributedMemoryMonitor created (standalone)")
        
        monitor.start_monitoring()
        print("   ✅ Monitoring started")
        
        time.sleep(3)  # 监控3秒
        
        monitor.stop_monitoring()
        print("   ✅ Monitoring stopped")
        
        # 测试获取内存分布
        memory_dist = monitor.get_memory_distribution()
        if 'error' not in memory_dist:
            print(f"   💾 Memory distribution: Node {memory_dist['node_id']}")
        else:
            print(f"   💾 Memory distribution: {memory_dist['error']}")
        
        # 测试生成报告
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            report = monitor.generate_distributed_report(f.name)
            print(f"   📄 Report generated: {f.name}")
            os.unlink(f.name)
        
        return True
        
    except Exception as e:
        print(f"   ❌ Distributed Memory Monitor test failed: {e}")
        return False

def test_cli_tool():
    """测试CLI工具"""
    print("\n🔍 Testing CLI Tool...")
    
    try:
        # 测试CLI工具是否可以正确导入和显示帮助
        result = subprocess.run([
            sys.executable, 'memory_debug_cli.py', '--help'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("   ✅ CLI tool help displayed successfully")
            return True
        else:
            print(f"   ❌ CLI tool failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("   ❌ CLI tool test timed out")
        return False
    except Exception as e:
        print(f"   ❌ CLI tool test failed: {e}")
        return False

def check_dependencies():
    """检查依赖项"""
    print("🔍 Checking dependencies...")
    
    dependencies = ['numpy', 'matplotlib', 'psutil']
    missing_deps = []
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"   ✅ {dep} available")
        except ImportError:
            print(f"   ❌ {dep} missing")
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("   Install with: pip install " + " ".join(missing_deps))
        return False
    
    return True

def check_nvidia_tools():
    """检查NVIDIA工具"""
    print("\n🔍 Checking NVIDIA tools...")
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("   ✅ nvidia-smi available")
            # 提取GPU信息
            lines = result.stdout.split('\n')
            gpu_count = 0
            for line in lines:
                if 'GeForce' in line or 'Tesla' in line or 'Quadro' in line or 'RTX' in line:
                    gpu_count += 1
            print(f"   📊 Found {gpu_count} GPU(s)")
            return True
        else:
            print("   ❌ nvidia-smi not working")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("   ❌ nvidia-smi not found")
        return False

def run_full_test():
    """运行完整测试套件"""
    print("🚀 Memory Debug Tools Test Suite")
    print("=" * 50)
    
    tests = [
        ("Dependencies", check_dependencies),
        ("NVIDIA Tools", check_nvidia_tools),
        ("Module Imports", test_imports),
        ("Memory Profiler", test_memory_profiler),
        ("OOM Detector", test_oom_detector),
        ("Fragmentation Analyzer", test_fragmentation_analyzer),
        ("Memory Leak Detector", test_memory_leak_detector),
        ("Distributed Monitor", test_distributed_monitor),
        ("CLI Tool", test_cli_tool)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"🎯 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Memory debug tools are working correctly.")
        return True
    else:
        print(f"⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_full_test()
    sys.exit(0 if success else 1)
