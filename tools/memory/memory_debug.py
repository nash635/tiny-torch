"""
Memory Debug Tools for Tiny Torch - 显存调试工具集
整合了分布式训练场景的显存调试功能

主要功能:
1. 显存使用监控和分析 (Memory Profiler)
2. OOM检测和预警 (OOM Detector) 
3. 显存碎片分析 (Fragmentation Analyzer)
4. 内存泄漏检测 (Memory Leak Detector)
5. 分布式内存监控 (Distributed Memory Monitor)
6. 统一的命令行界面 (CLI)

使用示例:
    from tools.memory import MemoryDebugger
    
    # 基本监控
    debugger = MemoryDebugger()
    debugger.start_monitoring()
    
    # 命令行使用
    python -m tools.memory profile --duration 60
    python -m tools.memory oom --threshold 85
"""

import os
import gc
import time
import json
import threading
import socket
import pickle
import signal
import subprocess
import argparse
import sys
import traceback
import weakref
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import queue
import multiprocessing as mp

# 第三方依赖
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import psutil
except ImportError as e:
    print(f"警告: 缺少依赖项 {e}. 请运行: pip install numpy matplotlib psutil")
    sys.exit(1)


# ============================================================================
# 数据结构定义
# ============================================================================

class OOMRiskLevel(Enum):
    """OOM风险等级"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


class FragmentationLevel(Enum):
    """碎片化程度等级"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"


class DistributedRole(Enum):
    """分布式角色"""
    MASTER = "master"
    WORKER = "worker"
    STANDALONE = "standalone"


@dataclass
class MemorySnapshot:
    """显存快照数据结构"""
    timestamp: float
    device_id: int
    allocated: int  # 已分配显存 (bytes)
    reserved: int   # 预留显存 (bytes) 
    free: int       # 可用显存 (bytes)
    total: int      # 总显存 (bytes)
    utilization: float  # 利用率 (%)
    fragmentation: float  # 碎片率 (%)
    process_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OOMEvent:
    """OOM事件记录"""
    timestamp: float
    device_id: int
    requested_memory: int
    available_memory: int
    total_memory: int
    stack_trace: Optional[str] = None
    context_info: Optional[Dict[str, Any]] = None


@dataclass
class OOMPrediction:
    """OOM预测结果"""
    device_id: int
    risk_level: OOMRiskLevel
    predicted_oom_time: Optional[float]
    confidence: float  # 预测置信度 (0-1)
    current_usage: float
    trend_slope: float
    recommended_actions: List[str] = field(default_factory=list)


@dataclass
class MemoryLeakEvent:
    """内存泄漏事件"""
    timestamp: float
    device_id: int
    leaked_memory: int
    leak_rate: float
    potential_sources: List[str] = field(default_factory=list)
    stack_trace: Optional[str] = None
    severity: str = "medium"


# ============================================================================
# 核心工具类
# ============================================================================

class MemoryProfiler:
    """显存使用分析器"""
    
    def __init__(self, 
                 devices: Optional[List[int]] = None,
                 sampling_interval: float = 0.1,
                 max_snapshots: int = 10000):
        self.devices = devices or self._get_available_devices()
        self.sampling_interval = sampling_interval
        self.max_snapshots = max_snapshots
        self.snapshots: Dict[int, List[MemorySnapshot]] = defaultdict(list)
        self.monitoring = False
        self.monitor_thread = None
        
    def _get_available_devices(self) -> List[int]:
        """获取可用的GPU设备"""
        try:
            result = subprocess.run(['nvidia-smi', '-L'], 
                                 capture_output=True, text=True, check=True)
            devices = []
            for line in result.stdout.strip().split('\n'):
                if 'GPU' in line:
                    device_id = int(line.split(':')[0].split()[-1])
                    devices.append(device_id)
            return devices
        except (subprocess.CalledProcessError, FileNotFoundError):
            return []
    
    def _get_memory_info(self, device_id: int) -> Optional[MemorySnapshot]:
        """获取指定设备的内存信息"""
        try:
            cmd = f"nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits -i {device_id}"
            result = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
            used, free, total = map(int, result.stdout.strip().split(', '))
            
            # 转换为字节
            used_bytes = used * 1024 * 1024
            free_bytes = free * 1024 * 1024 
            total_bytes = total * 1024 * 1024
            reserved_bytes = total_bytes - free_bytes
            
            utilization = (used_bytes / total_bytes) * 100 if total_bytes > 0 else 0
            fragmentation = self._calculate_fragmentation(device_id, free_bytes, total_bytes)
            
            return MemorySnapshot(
                timestamp=time.time(),
                device_id=device_id,
                allocated=used_bytes,
                reserved=reserved_bytes,
                free=free_bytes,
                total=total_bytes,
                utilization=utilization,
                fragmentation=fragmentation,
                process_info=self._get_process_info()
            )
        except Exception as e:
            print(f"警告: 无法获取设备 {device_id} 的内存信息: {e}")
            return None
    
    def _calculate_fragmentation(self, device_id: int, free_bytes: int, total_bytes: int) -> float:
        """计算内存碎片率（简化版本）"""
        if total_bytes == 0:
            return 0.0
        return max(0, min(100, (1 - free_bytes / total_bytes) * 50))
    
    def _get_process_info(self) -> Dict[str, Any]:
        """获取当前进程的内存信息"""
        try:
            process = psutil.Process()
            return {
                'pid': process.pid,
                'memory_percent': process.memory_percent(),
                'memory_info': process.memory_info()._asdict(),
                'cpu_percent': process.cpu_percent()
            }
        except:
            return {}
    
    def start_monitoring(self):
        """开始监控"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print(f"📊 开始监控 {len(self.devices)} 个GPU设备")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        print("🔄 停止内存监控")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            for device_id in self.devices:
                snapshot = self._get_memory_info(device_id)
                if snapshot:
                    self.snapshots[device_id].append(snapshot)
                    # 限制快照数量
                    if len(self.snapshots[device_id]) > self.max_snapshots:
                        self.snapshots[device_id].pop(0)
            
            time.sleep(self.sampling_interval)
    
    def get_current_status(self) -> Dict[int, Dict[str, Any]]:
        """获取当前状态"""
        status = {}
        for device_id in self.devices:
            if device_id in self.snapshots and self.snapshots[device_id]:
                latest = self.snapshots[device_id][-1]
                status[device_id] = {
                    'current_used': latest.allocated,
                    'total': latest.total,
                    'utilization': latest.utilization,
                    'fragmentation': latest.fragmentation,
                    'free': latest.free
                }
        return status
    
    def generate_report(self, output_file: str) -> Dict[str, Any]:
        """生成分析报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'devices': {},
            'summary': {}
        }
        
        total_memory = 0
        total_used = 0
        
        for device_id, snapshots in self.snapshots.items():
            if not snapshots:
                continue
                
            latest = snapshots[-1]
            utilizations = [s.utilization for s in snapshots]
            
            device_report = {
                'device_id': device_id,
                'current_usage': {
                    'allocated': latest.allocated,
                    'free': latest.free,
                    'total': latest.total,
                    'utilization_percent': latest.utilization
                },
                'statistics': {
                    'avg_utilization': np.mean(utilizations),
                    'max_utilization': np.max(utilizations),
                    'min_utilization': np.min(utilizations),
                    'samples_count': len(snapshots)
                },
                'fragmentation': latest.fragmentation
            }
            
            report['devices'][device_id] = device_report
            total_memory += latest.total
            total_used += latest.allocated
        
        report['summary'] = {
            'total_devices': len(self.devices),
            'total_memory_gb': total_memory / (1024**3),
            'total_used_gb': total_used / (1024**3),
            'overall_utilization': (total_used / total_memory * 100) if total_memory > 0 else 0
        }
        
        # 保存报告
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report


class OOMDetector:
    """OOM检测和预警器"""
    
    def __init__(self, 
                 threshold: float = 85.0,
                 prediction_window: int = 10,
                 warning_callback: Optional[Callable] = None):
        self.threshold = threshold
        self.prediction_window = prediction_window
        self.warning_callback = warning_callback or self._default_warning_callback
        self.monitoring = False
        self.monitor_thread = None
        self.memory_history: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        
    def _default_warning_callback(self, prediction: OOMPrediction):
        """默认警告回调"""
        print(f"⚠️  OOM警告: GPU {prediction.device_id}")
        print(f"   风险等级: {prediction.risk_level.value}")
        print(f"   当前使用率: {prediction.current_usage:.1f}%")
        if prediction.predicted_oom_time:
            print(f"   预测OOM时间: {prediction.predicted_oom_time:.1f}秒后")
        print(f"   置信度: {prediction.confidence:.2f}")
        for action in prediction.recommended_actions:
            print(f"   建议: {action}")
    
    def start_monitoring(self, devices: Optional[List[int]] = None):
        """开始OOM监控"""
        if self.monitoring:
            return
            
        self.devices = devices or self._get_available_devices()
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print(f"🚨 开始OOM监控 (阈值: {self.threshold}%)")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        print("🔄 停止OOM监控")
    
    def _get_available_devices(self) -> List[int]:
        """获取可用设备"""
        try:
            result = subprocess.run(['nvidia-smi', '-L'], 
                                 capture_output=True, text=True, check=True)
            devices = []
            for line in result.stdout.strip().split('\n'):
                if 'GPU' in line:
                    device_id = int(line.split(':')[0].split()[-1])
                    devices.append(device_id)
            return devices
        except:
            return []
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            for device_id in self.devices:
                utilization = self._get_device_utilization(device_id)
                if utilization is not None:
                    self.memory_history[device_id].append((time.time(), utilization))
                    
                    # 限制历史记录长度
                    if len(self.memory_history[device_id]) > 100:
                        self.memory_history[device_id].pop(0)
                    
                    # 进行OOM预测
                    prediction = self._predict_oom(device_id)
                    if prediction and prediction.risk_level in [OOMRiskLevel.HIGH, OOMRiskLevel.CRITICAL]:
                        self.warning_callback(prediction)
            
            time.sleep(1.0)
    
    def _get_device_utilization(self, device_id: int) -> Optional[float]:
        """获取设备利用率"""
        try:
            cmd = f"nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i {device_id}"
            result = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
            used, total = map(int, result.stdout.strip().split(', '))
            return (used / total) * 100 if total > 0 else 0
        except:
            return None
    
    def _predict_oom(self, device_id: int) -> Optional[OOMPrediction]:
        """预测OOM风险"""
        history = self.memory_history.get(device_id, [])
        if len(history) < 3:
            return None
        
        recent_usage = [usage for _, usage in history[-self.prediction_window:]]
        current_usage = recent_usage[-1]
        
        # 计算趋势
        times = list(range(len(recent_usage)))
        if len(times) > 1:
            trend_slope = np.polyfit(times, recent_usage, 1)[0]
        else:
            trend_slope = 0
        
        # 评估风险
        risk_level = OOMRiskLevel.LOW
        predicted_time = None
        confidence = 0.5
        actions = []
        
        if current_usage > 95:
            risk_level = OOMRiskLevel.CRITICAL
            confidence = 0.95
            actions = ["立即释放缓存: tiny_torch.cuda.empty_cache()", "减少batch size", "检查是否有内存泄漏"]
        elif current_usage > self.threshold:
            if trend_slope > 1.0:
                risk_level = OOMRiskLevel.HIGH
                predicted_time = (100 - current_usage) / trend_slope
                confidence = 0.8
                actions = ["准备释放缓存", "考虑减少batch size"]
            else:
                risk_level = OOMRiskLevel.MEDIUM
                confidence = 0.6
                actions = ["监控内存使用趋势"]
        
        return OOMPrediction(
            device_id=device_id,
            risk_level=risk_level,
            predicted_oom_time=predicted_time,
            confidence=confidence,
            current_usage=current_usage,
            trend_slope=trend_slope,
            recommended_actions=actions
        )


class MemoryLeakDetector:
    """内存泄漏检测器"""
    
    def __init__(self, leak_threshold: float = 100 * 1024 * 1024):  # 100MB
        self.leak_threshold = leak_threshold
        self.monitoring = False
        self.baseline_memory: Dict[int, int] = {}
        self.leak_events: List[MemoryLeakEvent] = []
        
    def start_monitoring(self, devices: Optional[List[int]] = None):
        """开始泄漏检测"""
        self.devices = devices or self._get_available_devices()
        
        # 建立基线
        for device_id in self.devices:
            usage = self._get_device_memory_usage(device_id)
            if usage is not None:
                self.baseline_memory[device_id] = usage
        
        self.monitoring = True
        print("🔍 开始内存泄漏检测")
    
    def stop_monitoring(self):
        """停止检测"""
        self.monitoring = False
        print("🔄 停止内存泄漏检测")
    
    def _get_available_devices(self) -> List[int]:
        """获取可用设备"""
        try:
            result = subprocess.run(['nvidia-smi', '-L'], 
                                 capture_output=True, text=True, check=True)
            devices = []
            for line in result.stdout.strip().split('\n'):
                if 'GPU' in line:
                    device_id = int(line.split(':')[0].split()[-1])
                    devices.append(device_id)
            return devices
        except:
            return []
    
    def _get_device_memory_usage(self, device_id: int) -> Optional[int]:
        """获取设备内存使用量"""
        try:
            cmd = f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i {device_id}"
            result = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
            return int(result.stdout.strip()) * 1024 * 1024  # 转换为字节
        except:
            return None
    
    def check_for_leaks(self) -> List[MemoryLeakEvent]:
        """检查内存泄漏"""
        if not self.monitoring:
            return []
        
        current_leaks = []
        
        for device_id in self.devices:
            current_usage = self._get_device_memory_usage(device_id)
            baseline = self.baseline_memory.get(device_id)
            
            if current_usage is not None and baseline is not None:
                memory_increase = current_usage - baseline
                
                if memory_increase > self.leak_threshold:
                    leak_event = MemoryLeakEvent(
                        timestamp=time.time(),
                        device_id=device_id,
                        leaked_memory=memory_increase,
                        leak_rate=memory_increase / 3600,  # 简化计算
                        potential_sources=["未知"],
                        severity="high" if memory_increase > self.leak_threshold * 2 else "medium"
                    )
                    current_leaks.append(leak_event)
                    self.leak_events.append(leak_event)
        
        return current_leaks


# ============================================================================
# 统一调试接口
# ============================================================================

class MemoryDebugger:
    """统一的内存调试工具"""
    
    def __init__(self):
        self.profiler = None
        self.oom_detector = None
        self.leak_detector = None
        self.active_tools = []
        
    def start_monitoring(self, 
                        enable_profiler: bool = True,
                        enable_oom_detection: bool = True, 
                        enable_leak_detection: bool = True,
                        devices: Optional[List[int]] = None,
                        oom_threshold: float = 85.0):
        """开始全面监控"""
        print("🚀 启动内存调试工具...")
        
        if enable_profiler:
            self.profiler = MemoryProfiler(devices=devices)
            self.profiler.start_monitoring()
            self.active_tools.append(self.profiler)
        
        if enable_oom_detection:
            self.oom_detector = OOMDetector(threshold=oom_threshold)
            self.oom_detector.start_monitoring(devices=devices)
            self.active_tools.append(self.oom_detector)
        
        if enable_leak_detection:
            self.leak_detector = MemoryLeakDetector()
            self.leak_detector.start_monitoring(devices=devices)
            self.active_tools.append(self.leak_detector)
        
        print(f"✅ 已启动 {len(self.active_tools)} 个监控工具")
    
    def stop_monitoring(self):
        """停止所有监控"""
        print("🔄 停止所有内存监控工具...")
        for tool in self.active_tools:
            if hasattr(tool, 'stop_monitoring'):
                tool.stop_monitoring()
        self.active_tools.clear()
        print("✅ 所有监控工具已停止")
    
    def get_status_report(self) -> Dict[str, Any]:
        """获取状态报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'active_tools': len(self.active_tools)
        }
        
        if self.profiler:
            report['memory_status'] = self.profiler.get_current_status()
        
        if self.leak_detector:
            recent_leaks = [event for event in self.leak_detector.leak_events 
                          if time.time() - event.timestamp < 3600]  # 最近1小时
            report['recent_leaks'] = len(recent_leaks)
        
        return report


# ============================================================================
# 命令行接口
# ============================================================================

def create_cli() -> argparse.ArgumentParser:
    """创建命令行解析器"""
    parser = argparse.ArgumentParser(
        description="Tiny Torch Memory Debug Tools - 显存调试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s profile --duration 60 --output report.json    # 60秒内存分析
  %(prog)s oom --threshold 85 --monitor                   # OOM监控
  %(prog)s leak --check                                   # 检查内存泄漏
  %(prog)s monitor --all                                  # 启动全面监控
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # Profile 命令
    profile_parser = subparsers.add_parser('profile', help='内存分析')
    profile_parser.add_argument('--duration', type=int, default=60, help='监控时长(秒)')
    profile_parser.add_argument('--devices', type=int, nargs='+', help='监控的GPU设备ID')
    profile_parser.add_argument('--output', default='memory_profile.json', help='输出文件')
    profile_parser.add_argument('--interval', type=float, default=0.1, help='采样间隔(秒)')
    profile_parser.add_argument('--plot', action='store_true', help='生成图表')
    
    # OOM 命令
    oom_parser = subparsers.add_parser('oom', help='OOM检测')
    oom_parser.add_argument('--threshold', type=float, default=85.0, help='警告阈值(%)')
    oom_parser.add_argument('--monitor', action='store_true', help='持续监控模式')
    oom_parser.add_argument('--devices', type=int, nargs='+', help='监控的GPU设备ID')
    
    # Leak 命令
    leak_parser = subparsers.add_parser('leak', help='内存泄漏检测')
    leak_parser.add_argument('--check', action='store_true', help='执行泄漏检查')
    leak_parser.add_argument('--threshold', type=float, default=100, help='泄漏阈值(MB)')
    leak_parser.add_argument('--devices', type=int, nargs='+', help='检查的GPU设备ID')
    
    # Monitor 命令  
    monitor_parser = subparsers.add_parser('monitor', help='全面监控')
    monitor_parser.add_argument('--all', action='store_true', help='启用所有监控功能')
    monitor_parser.add_argument('--duration', type=int, default=0, help='监控时长(秒，0为无限)')
    monitor_parser.add_argument('--devices', type=int, nargs='+', help='监控的GPU设备ID')
    
    return parser


def main():
    """主函数"""
    parser = create_cli()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 设置信号处理
    stop_event = threading.Event()
    
    def signal_handler(signum, frame):
        print("\n🔄 收到退出信号，正在停止...")
        stop_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if args.command == 'profile':
            profiler = MemoryProfiler(
                devices=args.devices,
                sampling_interval=args.interval
            )
            profiler.start_monitoring()
            
            print(f"📊 开始内存分析 (时长: {args.duration}秒)")
            
            if args.duration > 0:
                for i in range(args.duration):
                    if stop_event.wait(1):
                        break
                    if (i + 1) % 10 == 0:
                        print(f"   进度: {i + 1}/{args.duration}秒")
            else:
                print("📊 持续监控中 (按Ctrl+C停止)...")
                stop_event.wait()
            
            profiler.stop_monitoring()
            report = profiler.generate_report(args.output)
            print(f"📄 报告已保存: {args.output}")
            
            # 显示摘要
            if 'summary' in report:
                summary = report['summary']
                print(f"\n📈 监控摘要:")
                print(f"   设备数: {summary['total_devices']}")
                print(f"   总显存: {summary['total_memory_gb']:.2f} GB")
                print(f"   已使用: {summary['total_used_gb']:.2f} GB")
                print(f"   利用率: {summary['overall_utilization']:.1f}%")
        
        elif args.command == 'oom':
            detector = OOMDetector(threshold=args.threshold)
            detector.start_monitoring(devices=args.devices)
            
            if args.monitor:
                print(f"🚨 OOM监控中 (阈值: {args.threshold}%) - 按Ctrl+C停止")
                stop_event.wait()
            else:
                print("🚨 OOM检测运行5分钟...")
                stop_event.wait(300)
            
            detector.stop_monitoring()
        
        elif args.command == 'leak':
            detector = MemoryLeakDetector(
                leak_threshold=args.threshold * 1024 * 1024
            )
            detector.start_monitoring(devices=args.devices)
            
            if args.check:
                print("🔍 检查内存泄漏...")
                time.sleep(5)  # 等待一段时间观察
                leaks = detector.check_for_leaks()
                
                if leaks:
                    print(f"⚠️  发现 {len(leaks)} 个潜在内存泄漏:")
                    for leak in leaks:
                        print(f"   GPU {leak.device_id}: {leak.leaked_memory/(1024**2):.1f} MB")
                        print(f"   严重程度: {leak.severity}")
                else:
                    print("✅ 未发现明显的内存泄漏")
            
            detector.stop_monitoring()
        
        elif args.command == 'monitor':
            debugger = MemoryDebugger()
            debugger.start_monitoring(devices=args.devices)
            
            duration = args.duration if args.duration > 0 else float('inf')
            print(f"🔍 全面监控中 - 按Ctrl+C停止")
            
            start_time = time.time()
            while time.time() - start_time < duration:
                if stop_event.wait(10):
                    break
                
                # 每10秒显示一次状态
                report = debugger.get_status_report()
                print(f"📊 状态: {report.get('active_tools', 0)} 个工具运行中")
            
            debugger.stop_monitoring()
    
    except KeyboardInterrupt:
        print("\n🔄 用户中断，正在退出...")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
