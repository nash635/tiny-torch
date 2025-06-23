"""
Fragmentation Analyzer - 显存碎片分析工具
主要功能：
1. 显存碎片检测和分析
2. 碎片整理建议
3. 分配模式分析
4. 碎片化趋势预测
"""

import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import subprocess
import threading


class FragmentationLevel(Enum):
    """碎片化程度等级"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"


@dataclass
class MemoryBlock:
    """内存块信息"""
    start_address: int
    size: int
    is_allocated: bool
    allocation_time: Optional[float] = None
    block_id: Optional[str] = None


@dataclass
class FragmentationMetrics:
    """碎片化指标"""
    device_id: int
    timestamp: float
    total_memory: int
    allocated_memory: int
    free_memory: int
    largest_free_block: int
    num_free_blocks: int
    fragmentation_ratio: float  # 0-1, 越高越碎片化
    fragmentation_level: FragmentationLevel
    allocation_efficiency: float  # 分配效率
    defrag_benefit: float  # 碎片整理预期收益


@dataclass
class AllocationPattern:
    """分配模式分析"""
    average_allocation_size: int
    allocation_frequency: float  # 分配频率 (次/秒)
    deallocation_frequency: float  # 释放频率 (次/秒)
    size_distribution: Dict[str, int]  # 大小分布统计
    temporal_pattern: str  # 时间模式描述


class FragmentationAnalyzer:
    """显存碎片分析器"""
    
    def __init__(self, 
                 devices: Optional[List[int]] = None,
                 sampling_interval: float = 1.0,
                 max_history: int = 1000):
        """
        初始化碎片分析器
        
        Args:
            devices: 要监控的GPU设备ID列表
            sampling_interval: 采样间隔（秒）
            max_history: 最大历史记录数量
        """
        self.devices = devices or self._get_available_devices()
        self.sampling_interval = sampling_interval
        self.max_history = max_history
        
        # 碎片化历史数据
        self.fragmentation_history: Dict[int, List[FragmentationMetrics]] = {
            device_id: [] for device_id in self.devices
        }
        
        # 内存块跟踪（简化实现）
        self.memory_blocks: Dict[int, List[MemoryBlock]] = {
            device_id: [] for device_id in self.devices
        }
        
        # 分配事件记录
        self.allocation_events: Dict[int, List[Tuple[float, int, str]]] = {
            device_id: [] for device_id in self.devices
        }
        
        # 监控状态
        self.is_monitoring = False 
        self.monitor_thread = None
        
        # 阈值配置
        self.fragmentation_thresholds = {
            FragmentationLevel.LOW: 0.2,
            FragmentationLevel.MODERATE: 0.4,
            FragmentationLevel.HIGH: 0.6,
            FragmentationLevel.SEVERE: 0.8
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
    
    def _simulate_memory_blocks(self, device_id: int, mem_info: Dict[str, int]) -> List[MemoryBlock]:
        """
        模拟内存块分布（简化实现）
        实际应该通过CUDA memory allocator API获取真实的内存块信息
        """
        blocks = []
        
        # 模拟算法：基于历史分配模式生成内存块分布
        total_memory = mem_info['total']
        used_memory = mem_info['used']
        free_memory = mem_info['free']
        
        if total_memory == 0:
            return blocks
        
        # 模拟分配的内存块
        current_pos = 0
        remaining_used = used_memory
        
        # 基于经验的分配大小分布
        typical_sizes = [
            (64 * 1024 * 1024, 0.1),     # 64MB - 10%
            (128 * 1024 * 1024, 0.2),    # 128MB - 20%
            (256 * 1024 * 1024, 0.3),    # 256MB - 30%
            (512 * 1024 * 1024, 0.25),   # 512MB - 25%
            (1024 * 1024 * 1024, 0.15),  # 1GB - 15%
        ]
        
        block_id = 0
        while remaining_used > 0 and current_pos < total_memory:
            # 随机选择块大小
            size_idx = np.random.choice(len(typical_sizes), 
                                       p=[prob for _, prob in typical_sizes])
            block_size = min(typical_sizes[size_idx][0], remaining_used)
            
            if block_size > 0:
                blocks.append(MemoryBlock(
                    start_address=current_pos,
                    size=block_size,
                    is_allocated=True,
                    allocation_time=time.time() - np.random.uniform(0, 3600),  # 过去1小时内
                    block_id=f"block_{device_id}_{block_id}"
                ))
                
                current_pos += block_size
                remaining_used -= block_size
                block_id += 1
                
                # 添加一些空隙模拟碎片
                if np.random.random() < 0.3:  # 30%概率有空隙
                    gap_size = np.random.randint(1024*1024, 16*1024*1024)  # 1-16MB空隙
                    if current_pos + gap_size < total_memory:
                        blocks.append(MemoryBlock(
                            start_address=current_pos,
                            size=gap_size,
                            is_allocated=False
                        ))
                        current_pos += gap_size
        
        # 添加剩余的大块空闲内存
        if current_pos < total_memory:
            remaining_free = total_memory - current_pos
            blocks.append(MemoryBlock(
                start_address=current_pos,
                size=remaining_free,
                is_allocated=False
            ))
        
        return blocks
    
    def _calculate_fragmentation_metrics(self, device_id: int) -> FragmentationMetrics:
        """计算碎片化指标"""
        mem_info = self._get_memory_info(device_id)
        current_time = time.time()
        
        if mem_info['total'] == 0:
            return FragmentationMetrics(
                device_id=device_id,
                timestamp=current_time,
                total_memory=0,
                allocated_memory=0,
                free_memory=0,
                largest_free_block=0,
                num_free_blocks=0,
                fragmentation_ratio=0.0,
                fragmentation_level=FragmentationLevel.LOW,
                allocation_efficiency=1.0,
                defrag_benefit=0.0
            )
        
        # 获取或模拟内存块信息
        blocks = self._simulate_memory_blocks(device_id, mem_info)
        self.memory_blocks[device_id] = blocks
        
        # 分析空闲块
        free_blocks = [block for block in blocks if not block.is_allocated]
        free_block_sizes = [block.size for block in free_blocks]
        
        largest_free_block = max(free_block_sizes) if free_block_sizes else 0
        num_free_blocks = len(free_blocks)
        total_free = sum(free_block_sizes)
        
        # 计算碎片化比率
        if mem_info['free'] > 0:
            # 碎片化 = 1 - (最大空闲块 / 总空闲内存)
            fragmentation_ratio = 1.0 - (largest_free_block / mem_info['free'])
        else:
            fragmentation_ratio = 0.0
        
        # 确定碎片化等级
        fragmentation_level = FragmentationLevel.LOW
        for level, threshold in sorted(self.fragmentation_thresholds.items(), 
                                     key=lambda x: x[1], reverse=True):
            if fragmentation_ratio >= threshold:
                fragmentation_level = level
                break
        
        # 计算分配效率
        if mem_info['total'] > 0:
            allocation_efficiency = (mem_info['used'] / mem_info['total'])
        else:
            allocation_efficiency = 0.0
        
        # 估算碎片整理收益
        defrag_benefit = fragmentation_ratio * mem_info['free'] / mem_info['total']
        
        return FragmentationMetrics(
            device_id=device_id,
            timestamp=current_time,
            total_memory=mem_info['total'],
            allocated_memory=mem_info['used'],
            free_memory=mem_info['free'],
            largest_free_block=largest_free_block,
            num_free_blocks=num_free_blocks,
            fragmentation_ratio=fragmentation_ratio,
            fragmentation_level=fragmentation_level,
            allocation_efficiency=allocation_efficiency,
            defrag_benefit=defrag_benefit
        )
    
    def _analyze_allocation_pattern(self, device_id: int) -> AllocationPattern:
        """分析分配模式"""
        events = self.allocation_events[device_id]
        if not events:
            return AllocationPattern(
                average_allocation_size=0,
                allocation_frequency=0.0,
                deallocation_frequency=0.0,
                size_distribution={},
                temporal_pattern="No data"
            )
        
        # 分析最近1小时的数据
        current_time = time.time()
        recent_events = [
            (timestamp, size, event_type) for timestamp, size, event_type in events
            if current_time - timestamp <= 3600
        ]
        
        if not recent_events:
            return AllocationPattern(
                average_allocation_size=0,
                allocation_frequency=0.0,
                deallocation_frequency=0.0,
                size_distribution={},
                temporal_pattern="No recent data"
            )
        
        # 计算统计信息
        allocations = [(t, s) for t, s, e in recent_events if e == 'alloc']
        deallocations = [(t, s) for t, s, e in recent_events if e == 'free']
        
        avg_size = np.mean([s for _, s in allocations]) if allocations else 0
        alloc_freq = len(allocations) / 3600  # 每秒分配次数
        dealloc_freq = len(deallocations) / 3600  # 每秒释放次数
        
        # 大小分布统计
        size_ranges = {
            'small (<64MB)': 0,
            'medium (64MB-256MB)': 0,
            'large (256MB-1GB)': 0,
            'huge (>1GB)': 0
        }
        
        for _, size in allocations:
            if size < 64 * 1024 * 1024:
                size_ranges['small (<64MB)'] += 1
            elif size < 256 * 1024 * 1024:
                size_ranges['medium (64MB-256MB)'] += 1
            elif size < 1024 * 1024 * 1024:
                size_ranges['large (256MB-1GB)'] += 1
            else:
                size_ranges['huge (>1GB)'] += 1
        
        # 时间模式分析
        if alloc_freq > dealloc_freq * 1.5:
            temporal_pattern = "Memory accumulating"
        elif dealloc_freq > alloc_freq * 1.5:
            temporal_pattern = "Memory releasing"
        elif alloc_freq > 0.1:
            temporal_pattern = "High activity"
        else:
            temporal_pattern = "Low activity"
        
        return AllocationPattern(
            average_allocation_size=int(avg_size),
            allocation_frequency=alloc_freq,
            deallocation_frequency=dealloc_freq,
            size_distribution=size_ranges,
            temporal_pattern=temporal_pattern
        )
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                for device_id in self.devices:
                    # 计算碎片化指标
                    metrics = self._calculate_fragmentation_metrics(device_id)
                    
                    # 添加到历史记录
                    self.fragmentation_history[device_id].append(metrics)
                    
                    # 维护历史记录大小
                    if len(self.fragmentation_history[device_id]) > self.max_history:
                        self.fragmentation_history[device_id] = \
                            self.fragmentation_history[device_id][-self.max_history:]
                    
                    # 如果碎片化严重，输出警告
                    if metrics.fragmentation_level in [FragmentationLevel.HIGH, FragmentationLevel.SEVERE]:
                        print(f"⚠️  High fragmentation detected on GPU {device_id}: "
                              f"{metrics.fragmentation_level.value} "
                              f"({metrics.fragmentation_ratio:.2%})")
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                print(f"Error in fragmentation monitoring: {e}")
                time.sleep(5)
    
    def start_monitoring(self):
        """开始监控"""
        if self.is_monitoring:
            print("Fragmentation analyzer is already monitoring")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        print(f"Started fragmentation monitoring for devices: {self.devices}")
    
    def stop_monitoring(self):
        """停止监控"""
        if not self.is_monitoring:
            print("Fragmentation analyzer is not monitoring")
            return
        
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        print("Stopped fragmentation monitoring")
    
    def get_current_fragmentation(self) -> Dict[int, FragmentationMetrics]:
        """获取当前碎片化状态"""
        current_metrics = {}
        for device_id in self.devices:
            current_metrics[device_id] = self._calculate_fragmentation_metrics(device_id)
        return current_metrics
    
    def get_fragmentation_trend(self, device_id: int, duration_minutes: int = 30) -> Dict[str, Any]:
        """获取碎片化趋势"""
        if device_id not in self.devices:
            return {}
        
        current_time = time.time()
        cutoff_time = current_time - (duration_minutes * 60)
        
        # 筛选指定时间范围内的数据
        recent_metrics = [
            metrics for metrics in self.fragmentation_history[device_id]
            if metrics.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        # 计算趋势
        ratios = [m.fragmentation_ratio for m in recent_metrics]
        timestamps = [m.timestamp for m in recent_metrics]
        
        # 线性回归计算趋势
        if len(ratios) >= 2:
            coeffs = np.polyfit(timestamps, ratios, 1)
            trend_slope = coeffs[0]  # 每秒的变化率
        else:
            trend_slope = 0.0
        
        return {
            'device_id': device_id,
            'duration_minutes': duration_minutes,
            'data_points': len(recent_metrics),
            'current_fragmentation': ratios[-1] if ratios else 0,
            'min_fragmentation': min(ratios) if ratios else 0,
            'max_fragmentation': max(ratios) if ratios else 0,
            'avg_fragmentation': np.mean(ratios) if ratios else 0,
            'trend_slope': trend_slope,
            'trend_description': self._describe_trend(trend_slope)
        }
    
    def _describe_trend(self, slope: float) -> str:
        """描述趋势"""
        if abs(slope) < 1e-6:
            return "Stable"
        elif slope > 1e-4:
            return "Increasing fragmentation"
        elif slope < -1e-4:
            return "Decreasing fragmentation" 
        else:
            return "Minor fluctuation"
    
    def suggest_defragmentation(self, device_id: int) -> Dict[str, Any]:
        """生成碎片整理建议"""
        current_metrics = self._calculate_fragmentation_metrics(device_id)
        allocation_pattern = self._analyze_allocation_pattern(device_id)
        
        suggestions = {
            'device_id': device_id,
            'current_fragmentation_level': current_metrics.fragmentation_level.value,
            'fragmentation_ratio': current_metrics.fragmentation_ratio,
            'defrag_benefit': current_metrics.defrag_benefit,
            'recommendations': [],
            'priority': 'low'
        }
        
        # 基于碎片化程度生成建议
        if current_metrics.fragmentation_level == FragmentationLevel.SEVERE:
            suggestions['priority'] = 'urgent'
            suggestions['recommendations'].extend([
                "Immediate defragmentation required",
                "Consider torch.cuda.empty_cache()",
                "Restart training process if possible",
                "Reduce batch size to minimize large allocations"
            ])
        elif current_metrics.fragmentation_level == FragmentationLevel.HIGH:
            suggestions['priority'] = 'high'
            suggestions['recommendations'].extend([
                "Schedule defragmentation soon",
                "Use torch.cuda.empty_cache() at appropriate intervals",
                "Consider gradient accumulation to reduce peak memory"
            ])
        elif current_metrics.fragmentation_level == FragmentationLevel.MODERATE:
            suggestions['priority'] = 'medium'
            suggestions['recommendations'].extend([
                "Monitor fragmentation trend",
                "Optimize memory allocation patterns",
                "Consider memory-efficient optimizers"
            ])
        else:
            suggestions['recommendations'].append("Memory allocation is healthy")
        
        # 基于分配模式的建议
        if allocation_pattern.allocation_frequency > 10:  # 高频分配
            suggestions['recommendations'].append(
                "High allocation frequency detected - consider batch allocations"
            )
        
        if allocation_pattern.temporal_pattern == "Memory accumulating":
            suggestions['recommendations'].append(
                "Memory accumulation detected - check for memory leaks"
            )
        
        return suggestions
    
    def plot_fragmentation_timeline(self, device_id: int, output_file: Optional[str] = None):
        """绘制碎片化时间线图"""
        if device_id not in self.devices:
            print(f"Device {device_id} not being monitored")
            return
        
        metrics_history = self.fragmentation_history[device_id]
        if not metrics_history:
            print(f"No data available for device {device_id}")
            return
        
        # 准备数据
        timestamps = [(m.timestamp - metrics_history[0].timestamp) / 3600 for m in metrics_history]  # 转换为小时
        fragmentation_ratios = [m.fragmentation_ratio * 100 for m in metrics_history]  # 转换为百分比
        memory_usage = [m.allocated_memory / (1024**3) for m in metrics_history]  # 转换为GB
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 碎片化比率
        ax1.plot(timestamps, fragmentation_ratios, 'r-', linewidth=2, label='Fragmentation Ratio')
        ax1.axhline(y=20, color='orange', linestyle='--', alpha=0.7, label='Moderate Threshold')
        ax1.axhline(y=40, color='red', linestyle='--', alpha=0.7, label='High Threshold')
        ax1.set_ylabel('Fragmentation (%)')
        ax1.set_title(f'GPU {device_id} Memory Fragmentation Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # 内存使用量
        ax2.plot(timestamps, memory_usage, 'b-', linewidth=2, label='Memory Usage')
        ax2.set_ylabel('Memory Usage (GB)')
        ax2.set_xlabel('Time (hours)')
        ax2.set_title('Memory Usage')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Fragmentation timeline saved to {output_file}")
        else:
            plt.show()
    
    def generate_fragmentation_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """生成碎片化分析报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'devices': {},
            'summary': {}
        }
        
        fragmentation_levels = {level.value: 0 for level in FragmentationLevel}
        total_defrag_benefit = 0
        
        for device_id in self.devices:
            current_metrics = self._calculate_fragmentation_metrics(device_id)
            trend = self.get_fragmentation_trend(device_id)
            pattern = self._analyze_allocation_pattern(device_id)
            suggestions = self.suggest_defragmentation(device_id)
            
            device_report = {
                'device_id': device_id,
                'current_metrics': {
                    'fragmentation_level': current_metrics.fragmentation_level.value,
                    'fragmentation_ratio': current_metrics.fragmentation_ratio,
                    'num_free_blocks': current_metrics.num_free_blocks,
                    'largest_free_block_mb': current_metrics.largest_free_block / (1024*1024),
                    'allocation_efficiency': current_metrics.allocation_efficiency,
                    'defrag_benefit': current_metrics.defrag_benefit
                },
                'trend_analysis': trend,
                'allocation_pattern': {
                    'average_allocation_size_mb': pattern.average_allocation_size / (1024*1024),
                    'allocation_frequency': pattern.allocation_frequency,
                    'deallocation_frequency': pattern.deallocation_frequency,
                    'temporal_pattern': pattern.temporal_pattern,
                    'size_distribution': pattern.size_distribution
                },
                'recommendations': suggestions
            }
            
            report['devices'][device_id] = device_report
            fragmentation_levels[current_metrics.fragmentation_level.value] += 1
            total_defrag_benefit += current_metrics.defrag_benefit
        
        # 生成摘要
        report['summary'] = {
            'total_devices': len(self.devices),
            'fragmentation_distribution': fragmentation_levels,
            'total_defrag_benefit': total_defrag_benefit,
            'high_fragmentation_devices': [
                device_id for device_id in self.devices
                if self._calculate_fragmentation_metrics(device_id).fragmentation_level 
                in [FragmentationLevel.HIGH, FragmentationLevel.SEVERE]
            ]
        }
        
        # 保存报告
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Fragmentation analysis report saved to {output_file}")
        
        return report
    
    def simulate_allocation_event(self, device_id: int, size: int, event_type: str = 'alloc'):
        """模拟分配事件（用于测试）"""
        if device_id in self.devices:
            current_time = time.time()
            self.allocation_events[device_id].append((current_time, size, event_type))
            
            # 维护事件历史大小
            max_events = 1000
            if len(self.allocation_events[device_id]) > max_events:
                self.allocation_events[device_id] = self.allocation_events[device_id][-max_events:]


if __name__ == "__main__":
    # 演示用法
    analyzer = FragmentationAnalyzer(sampling_interval=2.0)
    
    try:
        analyzer.start_monitoring()
        
        # 模拟一些分配事件
        if analyzer.devices:
            test_device = analyzer.devices[0]
            print(f"Simulating allocation events on GPU {test_device}...")
            
            # 模拟各种大小的分配
            sizes = [64*1024*1024, 128*1024*1024, 256*1024*1024, 512*1024*1024]
            for i in range(20):
                size = np.random.choice(sizes)
                analyzer.simulate_allocation_event(test_device, size, 'alloc')
                time.sleep(0.1)
                
                # 偶尔释放内存
                if np.random.random() < 0.3:
                    analyzer.simulate_allocation_event(test_device, size, 'free')
        
        # 监控30秒
        time.sleep(30)
        
        # 获取当前碎片化状态
        current_frag = analyzer.get_current_fragmentation()
        print("\n=== Current Fragmentation Status ===")
        for device_id, metrics in current_frag.items():
            print(f"GPU {device_id}: {metrics.fragmentation_level.value} "
                  f"({metrics.fragmentation_ratio:.2%} fragmented)")
        
        # 生成碎片整理建议
        if current_frag:
            test_device = list(current_frag.keys())[0]
            suggestions = analyzer.suggest_defragmentation(test_device)
            print(f"\n=== Defragmentation Suggestions for GPU {test_device} ===")
            print(f"Priority: {suggestions['priority']}")
            for rec in suggestions['recommendations']:
                print(f"- {rec}")
        
        # 生成报告
        report = analyzer.generate_fragmentation_report("fragmentation_report.json")
        print(f"\nGenerated fragmentation report for {len(report['devices'])} devices")
        
    finally:
        analyzer.stop_monitoring()
