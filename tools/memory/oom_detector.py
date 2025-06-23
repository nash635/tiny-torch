"""
OOM Detector - 显存溢出检测和预警工具
主要功能：
1. OOM风险评估
2. 显存使用趋势预测
3. 自动OOM预警
4. OOM发生后的诊断分析
"""

import os
import time
import threading
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import subprocess
import warnings
import logging


class OOMRiskLevel(Enum):
    """OOM风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


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
    predicted_oom_time: Optional[float]  # 预测OOM发生时间（秒）
    confidence: float  # 预测置信度 (0-1)
    current_usage: float  # 当前使用率
    growth_rate: float  # 内存增长率 MB/s
    recommendation: str  # 建议操作


class OOMDetector:
    """OOM检测器主类"""
    
    def __init__(self, 
                 devices: Optional[List[int]] = None,
                 warning_threshold: float = 0.85,  # 85%显存使用率触发警告
                 critical_threshold: float = 0.95,  # 95%显存使用率触发严重警告
                 prediction_window: int = 30,  # 预测窗口（快照数量）
                 sampling_interval: float = 1.0):
        """
        初始化OOM检测器
        
        Args:
            devices: 要监控的GPU设备ID列表
            warning_threshold: 警告阈值（显存使用率）
            critical_threshold: 严重警告阈值（显存使用率）
            prediction_window: 用于预测的历史数据窗口大小
            sampling_interval: 采样间隔（秒）
        """
        self.devices = devices or self._get_available_devices()
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.prediction_window = prediction_window
        self.sampling_interval = sampling_interval
        
        # 历史数据存储
        self.memory_history: Dict[int, List[Tuple[float, int]]] = {
            device_id: [] for device_id in self.devices
        }
        
        # OOM事件记录
        self.oom_events: List[OOMEvent] = []
        
        # 回调函数
        self.warning_callbacks: List[Callable] = []
        self.oom_callbacks: List[Callable] = []
        
        # 监控状态
        self.is_monitoring = False
        self.monitor_thread = None
        
        # 设置日志
        self.logger = self._setup_logger()
    
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
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('OOMDetector')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
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
                    'used': used * 1024 * 1024,  # 转换为字节
                    'free': free * 1024 * 1024,
                    'total': total * 1024 * 1024
                }
        except Exception as e:
            self.logger.warning(f"Failed to get memory info for device {device_id}: {e}")
        
        return {'used': 0, 'free': 0, 'total': 0}
    
    def _calculate_growth_rate(self, device_id: int) -> float:
        """计算显存使用增长率 (MB/s)"""
        history = self.memory_history[device_id]
        if len(history) < 2:
            return 0.0
        
        # 使用最近的数据点计算斜率
        recent_points = history[-min(10, len(history)):]
        if len(recent_points) < 2:
            return 0.0
        
        # 线性回归计算增长率
        times = np.array([point[0] for point in recent_points])
        memories = np.array([point[1] for point in recent_points])
        
        if len(times) < 2 or np.std(times) == 0:
            return 0.0
        
        # 计算斜率 (bytes/second)
        coeffs = np.polyfit(times, memories, 1)
        growth_rate_bytes_per_sec = coeffs[0]
        
        # 转换为 MB/s
        return growth_rate_bytes_per_sec / (1024 * 1024)
    
    def _predict_oom_time(self, device_id: int) -> Optional[float]:
        """预测OOM发生时间（秒后）"""
        mem_info = self._get_memory_info(device_id)
        if mem_info['total'] == 0:
            return None
        
        growth_rate = self._calculate_growth_rate(device_id)
        if growth_rate <= 0:
            return None  # 内存使用没有增长趋势
        
        available_memory = mem_info['free']
        # 预测耗尽可用内存的时间
        time_to_oom = available_memory / (growth_rate * 1024 * 1024)  # 转换为秒
        
        return time_to_oom if time_to_oom > 0 else None
    
    def _assess_oom_risk(self, device_id: int) -> OOMPrediction:
        """评估OOM风险"""
        mem_info = self._get_memory_info(device_id)
        if mem_info['total'] == 0:
            return OOMPrediction(
                device_id=device_id,
                risk_level=OOMRiskLevel.LOW,
                predicted_oom_time=None,
                confidence=0.0,
                current_usage=0.0,
                growth_rate=0.0,
                recommendation="Device not available"
            )
        
        current_usage = mem_info['used'] / mem_info['total']
        growth_rate = self._calculate_growth_rate(device_id)
        predicted_oom_time = self._predict_oom_time(device_id)
        
        # 风险评估逻辑
        if current_usage >= self.critical_threshold:
            risk_level = OOMRiskLevel.CRITICAL
            confidence = 0.9
            recommendation = "Immediate action required! Free memory or reduce batch size"
        elif current_usage >= self.warning_threshold:
            if growth_rate > 0 and predicted_oom_time and predicted_oom_time < 60:
                risk_level = OOMRiskLevel.HIGH
                confidence = 0.8
                recommendation = "High risk! Consider reducing batch size or clearing cache"
            else:
                risk_level = OOMRiskLevel.MEDIUM
                confidence = 0.6
                recommendation = "Monitor closely, consider optimization"
        elif growth_rate > 10:  # 快速增长 (>10 MB/s)
            risk_level = OOMRiskLevel.MEDIUM
            confidence = 0.5
            recommendation = "Memory usage growing rapidly, monitor trend"
        else:
            risk_level = OOMRiskLevel.LOW
            confidence = 0.3
            recommendation = "Memory usage stable"
        
        # 调整置信度基于历史数据量
        history_factor = min(len(self.memory_history[device_id]) / self.prediction_window, 1.0)
        confidence *= history_factor
        
        return OOMPrediction(
            device_id=device_id,
            risk_level=risk_level,
            predicted_oom_time=predicted_oom_time,
            confidence=confidence,
            current_usage=current_usage,
            growth_rate=growth_rate,
            recommendation=recommendation
        )
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                for device_id in self.devices:
                    mem_info = self._get_memory_info(device_id)
                    current_time = time.time()
                    
                    # 记录历史数据
                    self.memory_history[device_id].append((current_time, mem_info['used']))
                    
                    # 维护历史数据窗口
                    max_history = self.prediction_window * 2
                    if len(self.memory_history[device_id]) > max_history:
                        self.memory_history[device_id] = self.memory_history[device_id][-max_history:]
                    
                    # 风险评估
                    prediction = self._assess_oom_risk(device_id)
                    
                    # 触发回调
                    if prediction.risk_level in [OOMRiskLevel.HIGH, OOMRiskLevel.CRITICAL]:
                        self._trigger_warning_callbacks(device_id, prediction)
                    
                    # 检测实际OOM（通过nvidia-smi错误输出）
                    self._check_for_oom_events(device_id)
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                self.logger.error(f"Error in OOM monitoring loop: {e}")
                time.sleep(5)  # 错误时等待更长时间
    
    def _check_for_oom_events(self, device_id: int):
        """检查是否发生了OOM事件"""
        # 这里简化实现，实际应该监控CUDA错误或系统日志
        # 可以通过监控进程是否异常退出、CUDA错误码等方式检测OOM
        pass
    
    def _trigger_warning_callbacks(self, device_id: int, prediction: OOMPrediction):
        """触发警告回调"""
        for callback in self.warning_callbacks:
            try:
                callback(device_id, prediction)
            except Exception as e:
                self.logger.error(f"Error in warning callback: {e}")
    
    def add_warning_callback(self, callback: Callable[[int, OOMPrediction], None]):
        """添加警告回调函数"""
        self.warning_callbacks.append(callback)
    
    def add_oom_callback(self, callback: Callable[[OOMEvent], None]):
        """添加OOM事件回调函数"""
        self.oom_callbacks.append(callback)
    
    def start_monitoring(self):
        """开始监控"""
        if self.is_monitoring:
            self.logger.info("OOM detector is already monitoring")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info(f"Started OOM monitoring for devices: {self.devices}")
    
    def stop_monitoring(self):
        """停止监控"""
        if not self.is_monitoring:
            self.logger.info("OOM detector is not monitoring")
            return
        
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        self.logger.info("Stopped OOM monitoring")
    
    def get_risk_assessment(self) -> Dict[int, OOMPrediction]:
        """获取所有设备的风险评估"""
        predictions = {}
        for device_id in self.devices:
            predictions[device_id] = self._assess_oom_risk(device_id)
        return predictions
    
    def get_memory_trends(self, device_id: int, duration_minutes: int = 10) -> Dict[str, Any]:
        """获取指定设备的内存使用趋势"""
        if device_id not in self.devices:
            return {}
        
        current_time = time.time()
        cutoff_time = current_time - (duration_minutes * 60)
        
        # 筛选指定时间范围内的数据
        recent_history = [
            (timestamp, memory) for timestamp, memory in self.memory_history[device_id]
            if timestamp >= cutoff_time
        ]
        
        if not recent_history:
            return {}
        
        timestamps = [item[0] for item in recent_history]
        memories = [item[1] for item in recent_history]
        
        return {
            'device_id': device_id,
            'start_time': min(timestamps),
            'end_time': max(timestamps),
            'min_memory': min(memories),
            'max_memory': max(memories),
            'avg_memory': np.mean(memories),
            'current_memory': memories[-1] if memories else 0,
            'growth_rate_mb_per_sec': self._calculate_growth_rate(device_id),
            'data_points': len(recent_history)
        }
    
    def generate_oom_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """生成OOM风险报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'devices': {},
            'risk_summary': {},
            'recommendations': []
        }
        
        risk_counts = {level.value: 0 for level in OOMRiskLevel}
        high_risk_devices = []
        
        for device_id in self.devices:
            prediction = self._assess_oom_risk(device_id)
            trends = self.get_memory_trends(device_id)
            
            device_report = {
                'device_id': device_id,
                'risk_assessment': {
                    'risk_level': prediction.risk_level.value,
                    'confidence': prediction.confidence,
                    'current_usage_percent': prediction.current_usage * 100,
                    'growth_rate_mb_per_sec': prediction.growth_rate,
                    'predicted_oom_time_seconds': prediction.predicted_oom_time,
                    'recommendation': prediction.recommendation
                },
                'memory_trends': trends
            }
            
            report['devices'][device_id] = device_report
            risk_counts[prediction.risk_level.value] += 1
            
            if prediction.risk_level in [OOMRiskLevel.HIGH, OOMRiskLevel.CRITICAL]:
                high_risk_devices.append(device_id)
        
        # 风险摘要
        report['risk_summary'] = {
            'total_devices': len(self.devices),
            'risk_distribution': risk_counts,
            'high_risk_devices': high_risk_devices,
            'monitoring_duration_hours': len(self.memory_history[self.devices[0]]) * self.sampling_interval / 3600 if self.devices else 0
        }
        
        # 全局建议
        if risk_counts['critical'] > 0:
            report['recommendations'].append("URGENT: Some devices are at critical OOM risk!")
        if risk_counts['high'] > 0:
            report['recommendations'].append("WARNING: Some devices are at high OOM risk")
        if sum(risk_counts.values()) == risk_counts['low']:
            report['recommendations'].append("All devices are operating normally")
        
        # 保存报告
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            self.logger.info(f"OOM risk report saved to {output_file}")
        
        return report
    
    def simulate_oom_scenario(self, device_id: int, memory_increase_mb: int = 100):
        """模拟OOM场景进行测试"""
        if device_id not in self.devices:
            self.logger.error(f"Device {device_id} not available for simulation")
            return
        
        self.logger.info(f"Simulating OOM scenario on device {device_id}")
        
        # 添加模拟的内存增长数据
        current_time = time.time()
        current_memory = self._get_memory_info(device_id)['used']
        
        # 模拟快速增长
        for i in range(10):
            simulated_memory = current_memory + (memory_increase_mb * 1024 * 1024 * (i + 1))
            self.memory_history[device_id].append((current_time + i, simulated_memory))
        
        # 评估风险
        prediction = self._assess_oom_risk(device_id)
        self.logger.info(f"Simulation result: Risk level = {prediction.risk_level.value}")
        
        return prediction


# 预定义的警告回调函数
def default_warning_callback(device_id: int, prediction: OOMPrediction):
    """默认的警告回调函数"""
    print(f"\n⚠️  OOM WARNING for GPU {device_id}")
    print(f"   Risk Level: {prediction.risk_level.value.upper()}")
    print(f"   Current Usage: {prediction.current_usage*100:.1f}%")
    print(f"   Growth Rate: {prediction.growth_rate:.2f} MB/s")
    if prediction.predicted_oom_time:
        print(f"   Predicted OOM in: {prediction.predicted_oom_time:.0f} seconds")
    print(f"   Recommendation: {prediction.recommendation}")
    print()

def email_warning_callback(device_id: int, prediction: OOMPrediction):
    """发送邮件警告的回调函数（示例）"""
    # 这里可以集成邮件发送功能
    print(f"📧 Would send email alert for GPU {device_id} OOM risk: {prediction.risk_level.value}")


if __name__ == "__main__":
    # 演示用法
    detector = OOMDetector(sampling_interval=2.0)
    
    # 添加回调函数
    detector.add_warning_callback(default_warning_callback)
    
    try:
        detector.start_monitoring()
        
        # 监控30秒
        time.sleep(30)
        
        # 获取风险评估
        risks = detector.get_risk_assessment()
        print("\n=== Current OOM Risk Assessment ===")
        for device_id, prediction in risks.items():
            print(f"GPU {device_id}: {prediction.risk_level.value} risk ({prediction.current_usage*100:.1f}% used)")
        
        # 生成报告
        report = detector.generate_oom_report("oom_risk_report.json")
        print(f"\nGenerated OOM risk report for {len(report['devices'])} devices")
        
        # 测试模拟功能
        if risks:
            test_device = list(risks.keys())[0]
            print(f"\nTesting OOM simulation on GPU {test_device}...")
            detector.simulate_oom_scenario(test_device, 500)  # 模拟500MB增长
        
    finally:
        detector.stop_monitoring()
