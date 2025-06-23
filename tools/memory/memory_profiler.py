"""
Memory Profiler - 显存使用分析工具
主要功能：
1. 实时显存监控
2. 显存使用峰值追踪
3. 内存分配/释放日志
4. 显存使用热图生成
"""

import os
import time
import threading
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import subprocess
import psutil


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
    process_info: Dict[str, Any]


class MemoryProfiler:
    """显存分析器主类"""
    
    def __init__(self, 
                 devices: Optional[List[int]] = None,
                 sampling_interval: float = 0.1,
                 max_snapshots: int = 10000):
        """
        初始化显存分析器
        
        Args:
            devices: 要监控的GPU设备ID列表，None表示监控所有设备
            sampling_interval: 采样间隔（秒）
            max_snapshots: 最大保存的快照数量
        """
        self.devices = devices or self._get_available_devices()
        self.sampling_interval = sampling_interval
        self.max_snapshots = max_snapshots
        
        self.snapshots: Dict[int, List[MemorySnapshot]] = {
            device_id: [] for device_id in self.devices
        }
        
        self.is_monitoring = False
        self.monitor_thread = None
        self.peak_memory: Dict[int, int] = {}
        self.allocation_events: List[Dict] = []
        
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
            # 使用nvidia-smi获取显存信息
            result = subprocess.run([
                'nvidia-smi', f'--query-gpu=memory.used,memory.free,memory.total',
                '--format=csv,noheader,nounits', f'--id={device_id}'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                used, free, total = map(int, result.stdout.strip().split(', '))
                # 转换为字节
                return {
                    'used': used * 1024 * 1024,
                    'free': free * 1024 * 1024, 
                    'total': total * 1024 * 1024
                }
        except Exception as e:
            print(f"Warning: Failed to get memory info for device {device_id}: {e}")
        
        return {'used': 0, 'free': 0, 'total': 0}
    
    def _calculate_fragmentation(self, device_id: int) -> float:
        """计算显存碎片率"""
        # 这里是简化实现，实际应该调用CUDA memory allocator API
        # 由于这是演示版本，我们使用启发式方法估算
        try:
            mem_info = self._get_memory_info(device_id)
            if mem_info['total'] == 0:
                return 0.0
            
            # 简化的碎片率计算：基于使用率的波动
            recent_snapshots = self.snapshots[device_id][-10:]
            if len(recent_snapshots) < 2:
                return 0.0
            
            utilizations = [s.utilization for s in recent_snapshots]
            variance = np.var(utilizations) if len(utilizations) > 1 else 0
            
            # 归一化碎片率 (0-1)
            return min(variance / 100.0, 1.0)
            
        except Exception:
            return 0.0
    
    def _get_process_info(self) -> Dict[str, Any]:
        """获取当前进程的内存使用信息"""
        try:
            process = psutil.Process()
            return {
                'pid': process.pid,
                'memory_percent': process.memory_percent(),
                'cpu_percent': process.cpu_percent(),
                'num_threads': process.num_threads()
            }
        except Exception:
            return {}
    
    def _take_snapshot(self, device_id: int) -> MemorySnapshot:
        """获取单个设备的显存快照"""
        mem_info = self._get_memory_info(device_id)
        fragmentation = self._calculate_fragmentation(device_id)
        
        utilization = (mem_info['used'] / mem_info['total'] * 100) if mem_info['total'] > 0 else 0
        
        return MemorySnapshot(
            timestamp=time.time(),
            device_id=device_id,
            allocated=mem_info['used'],
            reserved=0,  # 简化实现
            free=mem_info['free'],
            total=mem_info['total'],
            utilization=utilization,
            fragmentation=fragmentation,
            process_info=self._get_process_info()
        )
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                for device_id in self.devices:
                    snapshot = self._take_snapshot(device_id)
                    
                    # 添加快照
                    self.snapshots[device_id].append(snapshot)
                    
                    # 维护最大快照数量
                    if len(self.snapshots[device_id]) > self.max_snapshots:
                        self.snapshots[device_id] = self.snapshots[device_id][-self.max_snapshots:]
                    
                    # 更新峰值内存
                    if device_id not in self.peak_memory:
                        self.peak_memory[device_id] = snapshot.allocated
                    else:
                        self.peak_memory[device_id] = max(
                            self.peak_memory[device_id], 
                            snapshot.allocated
                        )
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(1)  # 错误时等待更长时间
    
    def start_monitoring(self):
        """开始监控"""
        if self.is_monitoring:
            print("Memory profiler is already monitoring")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        print(f"Started memory monitoring for devices: {self.devices}")
    
    def stop_monitoring(self):
        """停止监控"""
        if not self.is_monitoring:
            print("Memory profiler is not monitoring")
            return
        
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        print("Stopped memory monitoring")
    
    def get_current_status(self) -> Dict[int, Dict[str, Any]]:
        """获取当前显存状态"""
        status = {}
        for device_id in self.devices:
            if self.snapshots[device_id]:
                latest = self.snapshots[device_id][-1]
                status[device_id] = {
                    'current_used': latest.allocated,
                    'current_free': latest.free,
                    'total': latest.total,
                    'utilization': latest.utilization,
                    'fragmentation': latest.fragmentation,
                    'peak_memory': self.peak_memory.get(device_id, 0)
                }
        return status
    
    def generate_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """生成显存使用报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'devices': {},
            'summary': {}
        }
        
        total_peak = 0
        total_current = 0
        
        for device_id in self.devices:
            snapshots = self.snapshots[device_id]
            if not snapshots:
                continue
            
            latest = snapshots[-1]
            peak = self.peak_memory.get(device_id, 0)
            
            # 计算统计信息
            utilizations = [s.utilization for s in snapshots]
            fragmentations = [s.fragmentation for s in snapshots]
            
            device_report = {
                'device_id': device_id,
                'current_memory': {
                    'allocated': latest.allocated,
                    'free': latest.free,
                    'total': latest.total,
                    'utilization': latest.utilization
                },
                'peak_memory': peak,
                'statistics': {
                    'avg_utilization': np.mean(utilizations),
                    'max_utilization': np.max(utilizations),
                    'avg_fragmentation': np.mean(fragmentations),
                    'max_fragmentation': np.max(fragmentations),
                    'num_snapshots': len(snapshots)
                }
            }
            
            report['devices'][device_id] = device_report
            total_peak += peak
            total_current += latest.allocated
        
        # 总体摘要
        report['summary'] = {
            'total_devices': len(self.devices),
            'total_current_memory': total_current,
            'total_peak_memory': total_peak,
            'monitoring_duration': len(self.snapshots[self.devices[0]]) * self.sampling_interval if self.devices else 0
        }
        
        # 保存报告
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Memory report saved to {output_file}")
        
        return report
    
    def plot_memory_usage(self, device_id: int, output_file: Optional[str] = None):
        """绘制显存使用趋势图"""
        if device_id not in self.devices:
            print(f"Device {device_id} not being monitored")
            return
        
        snapshots = self.snapshots[device_id]
        if not snapshots:
            print(f"No data available for device {device_id}")
            return
        
        # 准备数据
        timestamps = [(s.timestamp - snapshots[0].timestamp) for s in snapshots]
        allocated = [s.allocated / (1024**3) for s in snapshots]  # 转换为GB
        utilizations = [s.utilization for s in snapshots]
        fragmentations = [s.fragmentation * 100 for s in snapshots]  # 转换为百分比
        
        # 创建图表
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # 显存使用量
        ax1.plot(timestamps, allocated, 'b-', linewidth=2)
        ax1.set_ylabel('Memory Usage (GB)')
        ax1.set_title(f'GPU {device_id} Memory Usage Over Time')
        ax1.grid(True, alpha=0.3)
        
        # 利用率
        ax2.plot(timestamps, utilizations, 'g-', linewidth=2)
        ax2.set_ylabel('Utilization (%)')
        ax2.set_title('Memory Utilization')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # 碎片率
        ax3.plot(timestamps, fragmentations, 'r-', linewidth=2)
        ax3.set_ylabel('Fragmentation (%)')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_title('Memory Fragmentation')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Memory usage plot saved to {output_file}")
        else:
            plt.show()
    
    def export_snapshots(self, output_file: str, device_id: Optional[int] = None):
        """导出快照数据"""
        if device_id is not None:
            devices_to_export = [device_id] if device_id in self.devices else []
        else:
            devices_to_export = self.devices
        
        export_data = {
            'metadata': {
                'export_time': datetime.now().isoformat(),
                'sampling_interval': self.sampling_interval,
                'devices': devices_to_export
            },
            'snapshots': {}
        }
        
        for dev_id in devices_to_export:
            snapshots_data = []
            for snapshot in self.snapshots[dev_id]:
                snapshots_data.append({
                    'timestamp': snapshot.timestamp,
                    'allocated': snapshot.allocated,
                    'free': snapshot.free,
                    'total': snapshot.total,
                    'utilization': snapshot.utilization,
                    'fragmentation': snapshot.fragmentation,
                    'process_info': snapshot.process_info
                })
            export_data['snapshots'][dev_id] = snapshots_data
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Snapshots exported to {output_file}")


if __name__ == "__main__":
    # 演示用法
    profiler = MemoryProfiler(sampling_interval=1.0)
    
    try:
        profiler.start_monitoring()
        
        # 监控10秒
        time.sleep(10)
        
        # 获取当前状态
        status = profiler.get_current_status()
        print("Current memory status:")
        for device_id, info in status.items():
            print(f"  GPU {device_id}: {info['utilization']:.1f}% used")
        
        # 生成报告
        report = profiler.generate_report("memory_report.json")
        print(f"Generated report for {len(report['devices'])} devices")
        
    finally:
        profiler.stop_monitoring()
