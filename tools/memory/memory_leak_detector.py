"""
Memory Leak Detector - 显存泄漏检测工具
主要功能：
1. 显存泄漏检测和定位
2. 内存使用趋势分析
3. 泄漏来源追踪
4. 自动清理建议
"""

import os
import gc
import time
import threading
import json
import numpy as np
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import subprocess
import traceback
import weakref


@dataclass
class MemoryLeakEvent:
    """内存泄漏事件"""
    timestamp: float
    device_id: int
    leaked_memory: int  # 泄漏的内存量（字节）
    leak_rate: float  # 泄漏率 (bytes/sec)
    potential_sources: List[str] = field(default_factory=list)
    stack_trace: Optional[str] = None
    severity: str = "medium"  # low, medium, high, critical


@dataclass
class MemorySnapshot:
    """内存快照"""
    timestamp: float
    device_id: int
    allocated_memory: int
    reserved_memory: int
    free_memory: int
    python_memory: int  # Python进程内存
    reference_count: int  # 对象引用计数
    gc_stats: Dict[str, int]  # 垃圾回收统计


class MemoryReference:
    """内存引用跟踪"""
    def __init__(self, obj_id: str, obj_type: str, size: int, stack_trace: str):
        self.obj_id = obj_id
        self.obj_type = obj_type
        self.size = size
        self.creation_time = time.time()
        self.stack_trace = stack_trace
        self.last_access_time = time.time()


class MemoryLeakDetector:
    """显存泄漏检测器"""
    
    def __init__(self, 
                 devices: Optional[List[int]] = None,
                 sampling_interval: float = 5.0,
                 leak_threshold: float = 1024*1024*10,  # 10MB/min 泄漏阈值
                 history_window: int = 100,
                 enable_reference_tracking: bool = True):
        """
        初始化泄漏检测器
        
        Args:
            devices: 要监控的GPU设备ID列表
            sampling_interval: 采样间隔（秒）
            leak_threshold: 泄漏阈值（字节/分钟）
            history_window: 历史数据窗口大小
            enable_reference_tracking: 是否启用对象引用跟踪
        """
        self.devices = devices or self._get_available_devices()
        self.sampling_interval = sampling_interval
        self.leak_threshold = leak_threshold
        self.history_window = history_window
        self.enable_reference_tracking = enable_reference_tracking
        
        # 历史数据存储
        self.memory_snapshots: Dict[int, List[MemorySnapshot]] = {
            device_id: [] for device_id in self.devices
        }
        
        # 泄漏事件记录
        self.leak_events: List[MemoryLeakEvent] = []
        
        # 对象引用跟踪
        self.tracked_references: Dict[str, MemoryReference] = {}
        self.reference_creation_counts: Dict[str, int] = defaultdict(int)
        
        # 基线内存使用量
        self.baseline_memory: Dict[int, int] = {}
        
        # 监控状态
        self.is_monitoring = False
        self.monitor_thread = None
        
        # 统计信息
        self.stats = {
            'total_leaks_detected': 0,
            'total_leaked_memory': 0,
            'leak_detection_start_time': None
        }
    
    def _get_available_devices(self) -> List[int]:
        """获取可用的GPU设备列表"""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=index', '--format=csv,noheader'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                return [int(line.strip()) for line in result.stdout.strip().split('\n')]
            return []
        except Exception:
            return []
    
    def _get_memory_info(self, device_id: int) -> Dict[str, int]:
        """获取指定设备的显存信息"""
        try:
            result = subprocess.run([
                'nvidia-smi', f'--query-gpu=memory.used,memory.free,memory.total',
                '--format=csv,noheader,nounits', f'--id={device_id}'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                used, free, total = map(int, result.stdout.strip().split(', '))
                return {
                    'used': used * 1024 * 1024,
                    'free': free * 1024 * 1024,
                    'total': total * 1024 * 1024
                }
        except Exception:
            pass
        
        return {'used': 0, 'free': 0, 'total': 0}
    
    def _get_python_memory_info(self) -> Dict[str, Any]:
        """获取Python进程内存信息"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss': memory_info.rss,  # 物理内存
                'vms': memory_info.vms,  # 虚拟内存
                'percent': process.memory_percent()
            }
        except Exception:
            return {'rss': 0, 'vms': 0, 'percent': 0.0}
    
    def _get_gc_stats(self) -> Dict[str, int]:
        """获取垃圾回收统计信息"""
        try:
            stats = gc.get_stats()
            total_collections = sum(gen['collections'] for gen in stats)
            total_collected = sum(gen['collected'] for gen in stats)
            total_uncollectable = sum(gen['uncollectable'] for gen in stats)
            
            return {
                'total_objects': len(gc.get_objects()),
                'total_collections': total_collections,
                'total_collected': total_collected,
                'total_uncollectable': total_uncollectable,
                'referrers_count': len(gc.get_referrers())
            }
        except Exception:
            return {}
    
    def _take_memory_snapshot(self, device_id: int) -> MemorySnapshot:
        """获取内存快照"""
        gpu_memory = self._get_memory_info(device_id)
        python_memory = self._get_python_memory_info()
        gc_stats = self._get_gc_stats()
        
        return MemorySnapshot(
            timestamp=time.time(),
            device_id=device_id,
            allocated_memory=gpu_memory['used'],
            reserved_memory=0,  # 简化实现
            free_memory=gpu_memory['free'],
            python_memory=python_memory['rss'],
            reference_count=gc_stats.get('total_objects', 0),
            gc_stats=gc_stats
        )
    
    def _detect_memory_leak(self, device_id: int) -> Optional[MemoryLeakEvent]:
        """检测内存泄漏"""
        snapshots = self.memory_snapshots[device_id]
        if len(snapshots) < 10:  # 需要足够的数据点
            return None
        
        # 分析最近的内存使用趋势
        recent_snapshots = snapshots[-10:]
        timestamps = [s.timestamp for s in recent_snapshots]
        allocated_memories = [s.allocated_memory for s in recent_snapshots]
        
        # 计算内存增长率 (线性回归)
        if len(timestamps) >= 2 and np.std(timestamps) > 0:
            coeffs = np.polyfit(timestamps, allocated_memories, 1)
            growth_rate = coeffs[0]  # bytes/second
            
            # 转换为 bytes/minute
            growth_rate_per_minute = growth_rate * 60
            
            # 检查是否超过泄漏阈值
            if growth_rate_per_minute > self.leak_threshold:
                # 计算泄漏的内存量
                current_memory = recent_snapshots[-1].allocated_memory
                baseline = self.baseline_memory.get(device_id, recent_snapshots[0].allocated_memory)
                leaked_memory = max(0, current_memory - baseline)
                
                # 确定严重性
                if growth_rate_per_minute > self.leak_threshold * 5:
                    severity = "critical"
                elif growth_rate_per_minute > self.leak_threshold * 2:
                    severity = "high"
                else:
                    severity = "medium"
                
                # 分析潜在来源
                potential_sources = self._analyze_leak_sources(device_id)
                
                return MemoryLeakEvent(
                    timestamp=time.time(),
                    device_id=device_id,
                    leaked_memory=leaked_memory,
                    leak_rate=growth_rate_per_minute,
                    potential_sources=potential_sources,
                    stack_trace=self._get_current_stack_trace(),
                    severity=severity
                )
        
        return None
    
    def _analyze_leak_sources(self, device_id: int) -> List[str]:
        """分析潜在的泄漏来源"""
        sources = []
        
        # 分析对象引用计数变化
        if self.enable_reference_tracking:
            gc_stats = self._get_gc_stats()
            recent_snapshots = self.memory_snapshots[device_id][-5:]
            
            if len(recent_snapshots) >= 2:
                ref_count_change = (recent_snapshots[-1].reference_count - 
                                  recent_snapshots[0].reference_count)
                
                if ref_count_change > 1000:
                    sources.append("Rapid object creation (possible circular references)")
            
            # 分析垃圾回收效率
            if gc_stats.get('total_uncollectable', 0) > 100:
                sources.append("High uncollectable objects count")
        
        # 分析内存分配模式
        recent_memories = [s.allocated_memory for s in self.memory_snapshots[device_id][-10:]]
        if recent_memories:
            memory_variance = np.var(recent_memories)
            if memory_variance > (1024*1024*100)**2:  # 高方差
                sources.append("Irregular memory allocation pattern")
        
        # 检查Python内存增长
        python_memories = [s.python_memory for s in self.memory_snapshots[device_id][-5:]]
        if len(python_memories) >= 2:
            python_growth = python_memories[-1] - python_memories[0]
            if python_growth > 50*1024*1024:  # 50MB
                sources.append("Python process memory growth")
        
        if not sources:
            sources.append("Unknown source - requires detailed profiling")
        
        return sources
    
    def _get_current_stack_trace(self) -> str:
        """获取当前调用栈"""
        try:
            return traceback.format_stack()[-5:]  # 最近5层调用栈
        except Exception:
            return "Stack trace not available"
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                for device_id in self.devices:
                    # 获取内存快照
                    snapshot = self._take_memory_snapshot(device_id)
                    self.memory_snapshots[device_id].append(snapshot)
                    
                    # 维护历史数据窗口
                    if len(self.memory_snapshots[device_id]) > self.history_window:
                        self.memory_snapshots[device_id] = \
                            self.memory_snapshots[device_id][-self.history_window:]
                    
                    # 检测内存泄漏
                    leak_event = self._detect_memory_leak(device_id)
                    if leak_event:
                        self.leak_events.append(leak_event)
                        self.stats['total_leaks_detected'] += 1
                        self.stats['total_leaked_memory'] += leak_event.leaked_memory
                        
                        # 输出警告
                        print(f"\n🚨 MEMORY LEAK DETECTED on GPU {device_id}")
                        print(f"   Severity: {leak_event.severity.upper()}")
                        print(f"   Leak Rate: {leak_event.leak_rate/(1024*1024):.2f} MB/min")
                        print(f"   Leaked Memory: {leak_event.leaked_memory/(1024*1024):.2f} MB")
                        print(f"   Potential Sources: {', '.join(leak_event.potential_sources)}")
                        print()
                
                # 定期触发垃圾回收
                if self.enable_reference_tracking:
                    gc.collect()
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                print(f"Error in memory leak monitoring: {e}")
                time.sleep(10)
    
    def start_monitoring(self):
        """开始监控"""
        if self.is_monitoring:
            print("Memory leak detector is already monitoring")
            return
        
        # 设置基线内存
        for device_id in self.devices:
            memory_info = self._get_memory_info(device_id)
            self.baseline_memory[device_id] = memory_info['used']
        
        self.is_monitoring = True
        self.stats['leak_detection_start_time'] = time.time()
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        print(f"Started memory leak monitoring for devices: {self.devices}")
        print(f"Leak threshold: {self.leak_threshold/(1024*1024):.1f} MB/min")
    
    def stop_monitoring(self):
        """停止监控"""
        if not self.is_monitoring:
            print("Memory leak detector is not monitoring")
            return
        
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=3.0)
        print("Stopped memory leak monitoring")
    
    def set_baseline(self, device_id: Optional[int] = None):
        """设置内存使用基线"""
        devices_to_set = [device_id] if device_id is not None else self.devices
        
        for dev_id in devices_to_set:
            if dev_id in self.devices:
                memory_info = self._get_memory_info(dev_id)
                self.baseline_memory[dev_id] = memory_info['used']
                print(f"Set baseline for GPU {dev_id}: {memory_info['used']/(1024*1024):.2f} MB")
    
    def force_garbage_collection(self):
        """强制执行垃圾回收"""
        print("Forcing garbage collection...")
        
        # Python垃圾回收
        collected = gc.collect()
        print(f"Python GC collected {collected} objects")
        
        # 尝试清理CUDA缓存（如果可用）
        try:
            # 这里应该调用 torch.cuda.empty_cache() 
            # 由于这是演示代码，我们用打印代替
            print("Clearing CUDA cache...")
            # torch.cuda.empty_cache()
        except Exception as e:
            print(f"Could not clear CUDA cache: {e}")
    
    def get_leak_summary(self) -> Dict[str, Any]:
        """获取泄漏检测摘要"""
        if not self.stats['leak_detection_start_time']:
            return {'error': 'Monitoring not started'}
        
        monitoring_duration = time.time() - self.stats['leak_detection_start_time']
        
        # 按设备统计泄漏
        leaks_by_device = defaultdict(list)
        for leak in self.leak_events:
            leaks_by_device[leak.device_id].append(leak)
        
        # 按严重性统计
        severity_counts = defaultdict(int)
        for leak in self.leak_events:
            severity_counts[leak.severity] += 1
        
        return {
            'monitoring_duration_hours': monitoring_duration / 3600,
            'total_leaks_detected': self.stats['total_leaks_detected'],
            'total_leaked_memory_mb': self.stats['total_leaked_memory'] / (1024*1024),
            'leaks_by_device': {
                device_id: len(leaks) for device_id, leaks in leaks_by_device.items()
            },
            'severity_distribution': dict(severity_counts),
            'leak_rate_per_hour': self.stats['total_leaks_detected'] / (monitoring_duration / 3600) if monitoring_duration > 0 else 0,
            'recent_leak_events': [
                {
                    'timestamp': leak.timestamp,
                    'device_id': leak.device_id,
                    'severity': leak.severity,
                    'leaked_memory_mb': leak.leaked_memory / (1024*1024),
                    'leak_rate_mb_per_min': leak.leak_rate / (1024*1024)
                }
                for leak in self.leak_events[-5:]  # 最近5个事件
            ]
        }
    
    def get_memory_growth_trend(self, device_id: int, duration_minutes: int = 30) -> Dict[str, Any]:
        """获取内存增长趋势"""
        if device_id not in self.devices:
            return {}
        
        current_time = time.time()
        cutoff_time = current_time - (duration_minutes * 60)
        
        # 筛选指定时间范围内的数据
        recent_snapshots = [
            snapshot for snapshot in self.memory_snapshots[device_id]
            if snapshot.timestamp >= cutoff_time
        ]
        
        if len(recent_snapshots) < 2:
            return {'error': 'Insufficient data'}
        
        # 计算趋势
        timestamps = [s.timestamp for s in recent_snapshots]
        gpu_memories = [s.allocated_memory for s in recent_snapshots]
        python_memories = [s.python_memory for s in recent_snapshots]
        
        # GPU内存趋势
        gpu_coeffs = np.polyfit(timestamps, gpu_memories, 1)
        gpu_growth_rate = gpu_coeffs[0] * 60  # bytes/minute
        
        # Python内存趋势
        python_coeffs = np.polyfit(timestamps, python_memories, 1)
        python_growth_rate = python_coeffs[0] * 60  # bytes/minute
        
        return {
            'device_id': device_id,
            'duration_minutes': duration_minutes,
            'data_points': len(recent_snapshots),
            'gpu_memory': {
                'start_mb': gpu_memories[0] / (1024*1024),
                'end_mb': gpu_memories[-1] / (1024*1024),
                'growth_rate_mb_per_min': gpu_growth_rate / (1024*1024),
                'total_growth_mb': (gpu_memories[-1] - gpu_memories[0]) / (1024*1024)
            },
            'python_memory': {
                'start_mb': python_memories[0] / (1024*1024),
                'end_mb': python_memories[-1] / (1024*1024),
                'growth_rate_mb_per_min': python_growth_rate / (1024*1024),
                'total_growth_mb': (python_memories[-1] - python_memories[0]) / (1024*1024)
            },
            'leak_risk': 'high' if gpu_growth_rate > self.leak_threshold else 'low'
        }
    
    def generate_leak_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """生成泄漏检测报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'monitoring_summary': self.get_leak_summary(),
            'devices': {},
            'leak_events': [],
            'recommendations': []
        }
        
        # 各设备详细信息
        for device_id in self.devices:
            trend = self.get_memory_growth_trend(device_id)
            if 'error' not in trend:
                report['devices'][device_id] = trend
        
        # 泄漏事件详情
        for leak in self.leak_events[-20:]:  # 最近20个事件
            report['leak_events'].append({
                'timestamp': datetime.fromtimestamp(leak.timestamp).isoformat(),
                'device_id': leak.device_id,
                'severity': leak.severity,
                'leaked_memory_mb': leak.leaked_memory / (1024*1024),
                'leak_rate_mb_per_min': leak.leak_rate / (1024*1024),
                'potential_sources': leak.potential_sources
            })
        
        # 生成建议
        summary = report['monitoring_summary']
        if isinstance(summary, dict) and 'total_leaks_detected' in summary:
            if summary['total_leaks_detected'] == 0:
                report['recommendations'].append("No memory leaks detected - system appears healthy")
            elif summary['total_leaks_detected'] < 5:
                report['recommendations'].append("Minor memory leaks detected - monitor closely")
            else:
                report['recommendations'].extend([
                    "Multiple memory leaks detected - immediate action recommended",
                    "Consider reducing batch sizes or model complexity",
                    "Review code for circular references or unclosed resources",
                    "Schedule regular garbage collection"
                ])
        
        # 保存报告
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Memory leak report saved to {output_file}")
        
        return report
    
    def suggest_fixes(self, device_id: int) -> List[str]:
        """提供修复建议"""
        suggestions = []
        
        # 分析该设备的泄漏情况
        device_leaks = [leak for leak in self.leak_events if leak.device_id == device_id]
        
        if not device_leaks:
            suggestions.append("No leaks detected for this device")
            return suggestions
        
        recent_leaks = device_leaks[-5:]  # 最近5个泄漏事件
        
        # 基于泄漏源分析生成建议
        all_sources = []
        for leak in recent_leaks:
            all_sources.extend(leak.potential_sources)
        
        source_counts = defaultdict(int)
        for source in all_sources:
            source_counts[source] += 1
        
        # 基于最常见的泄漏源生成建议
        most_common_source = max(source_counts.items(), key=lambda x: x[1])[0] if source_counts else ""
        
        if "circular references" in most_common_source.lower():
            suggestions.extend([
                "Break circular references by using weak references",
                "Review object lifecycle management",
                "Use context managers for resource cleanup"
            ])
        elif "uncollectable objects" in most_common_source.lower():
            suggestions.extend([
                "Enable Python garbage collection debugging",
                "Review C extension usage",
                "Check for __del__ method issues"
            ])
        elif "python process memory" in most_common_source.lower():
            suggestions.extend([
                "Monitor Python object creation patterns",
                "Consider using memory profiling tools",
                "Review data structure choices"
            ])
        else:
            suggestions.extend([
                "Enable detailed memory profiling",
                "Use torch.cuda.memory_stats() for detailed analysis",
                "Consider reducing model size or batch size",
                "Schedule regular cache clearing"
            ])
        
        # 通用建议
        suggestions.extend([
            "Monitor memory usage patterns over time",
            "Consider using memory-efficient alternatives",
            "Implement regular cleanup routines"
        ])
        
        return suggestions


if __name__ == "__main__":
    # 演示用法
    detector = MemoryLeakDetector(
        sampling_interval=3.0,
        leak_threshold=5*1024*1024,  # 5MB/min threshold for demo
        enable_reference_tracking=True
    )
    
    try:
        detector.start_monitoring()
        
        # 模拟一些内存使用
        print("Simulating memory usage patterns...")
        memory_hog = []
        
        for i in range(20):
            # 模拟内存增长
            memory_hog.extend([0] * (100000 + i * 10000))  # 逐渐增长的内存使用
            time.sleep(2)
            
            if i % 5 == 0:
                print(f"Step {i}: Allocated ~{len(memory_hog) * 8 / 1024 / 1024:.1f} MB")
        
        # 监控一段时间
        time.sleep(30)
        
        # 获取泄漏摘要
        summary = detector.get_leak_summary()
        print("\n=== Memory Leak Detection Summary ===")
        if 'error' not in summary:
            print(f"Monitoring Duration: {summary['monitoring_duration_hours']:.2f} hours")
            print(f"Total Leaks Detected: {summary['total_leaks_detected']}")
            print(f"Total Leaked Memory: {summary['total_leaked_memory_mb']:.2f} MB")
            print(f"Severity Distribution: {summary['severity_distribution']}")
        
        # 生成详细报告
        report = detector.generate_leak_report("memory_leak_report.json")
        print(f"\nGenerated memory leak report")
        
        # 获取修复建议
        if detector.devices:
            test_device = detector.devices[0]
            suggestions = detector.suggest_fixes(test_device)
            print(f"\n=== Fix Suggestions for GPU {test_device} ===")
            for suggestion in suggestions:
                print(f"- {suggestion}")
        
        # 强制垃圾回收
        detector.force_garbage_collection()
        
    finally:
        detector.stop_monitoring()
