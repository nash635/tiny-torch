"""
Distributed Memory Monitor - 分布式训练显存监控工具
主要功能：
1. 多GPU显存使用协调监控
2. 分布式训练显存平衡分析
3. 跨节点显存同步状态检测
4. 分布式OOM预防和恢复
"""

import os
import time
import json
import threading
import socket
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import subprocess
import numpy as np
from collections import defaultdict
import queue
import multiprocessing as mp


class DistributedRole(Enum):
    """分布式角色"""
    MASTER = "master"
    WORKER = "worker"
    STANDALONE = "standalone"


class SyncStatus(Enum):
    """同步状态"""
    IN_SYNC = "in_sync"
    OUT_OF_SYNC = "out_of_sync"
    COMMUNICATION_ERROR = "communication_error"
    UNKNOWN = "unknown"


@dataclass
class NodeInfo:
    """节点信息"""
    node_id: str
    hostname: str
    ip_address: str
    port: int
    role: DistributedRole
    gpu_devices: List[int]
    last_heartbeat: float
    is_active: bool = True


@dataclass
class DistributedMemoryState:
    """分布式内存状态"""
    timestamp: float
    global_step: int
    nodes: Dict[str, Dict[int, Dict[str, Any]]]  # node_id -> device_id -> memory_info
    memory_balance_ratio: float  # 内存平衡比率
    sync_status: SyncStatus
    bottleneck_nodes: List[str]  # 瓶颈节点
    total_memory_usage: int
    recommendations: List[str] = field(default_factory=list)


@dataclass
class DistributedOOMEvent:
    """分布式OOM事件"""
    timestamp: float
    node_id: str
    device_id: int
    global_step: int
    oom_type: str  # 'single_gpu', 'node_level', 'cluster_level'
    affected_nodes: List[str]
    recovery_actions: List[str] = field(default_factory=list)


class DistributedMemoryMonitor:
    """分布式显存监控器"""
    
    def __init__(self,
                 node_id: Optional[str] = None,
                 role: DistributedRole = DistributedRole.STANDALONE,
                 master_address: Optional[str] = None,
                 master_port: int = 29500,
                 devices: Optional[List[int]] = None,
                 sampling_interval: float = 2.0,
                 heartbeat_interval: float = 10.0,
                 sync_timeout: float = 30.0):
        """
        初始化分布式内存监控器
        
        Args:
            node_id: 节点ID，None时自动生成
            role: 节点角色
            master_address: master节点地址
            master_port: master节点端口
            devices: 要监控的GPU设备列表
            sampling_interval: 采样间隔（秒）
            heartbeat_interval: 心跳间隔（秒） 
            sync_timeout: 同步超时时间（秒）
        """
        # 网络配置 (需要在生成node_id之前设置)
        self.hostname = socket.gethostname()
        self.ip_address = self._get_local_ip()
        
        self.node_id = node_id or self._generate_node_id()
        self.role = role
        self.master_address = master_address
        self.master_port = master_port
        self.devices = devices or self._get_available_devices()
        self.sampling_interval = sampling_interval
        self.heartbeat_interval = heartbeat_interval
        self.sync_timeout = sync_timeout
        self.server_socket = None
        self.client_connections: Dict[str, socket.socket] = {}
        
        # 节点注册表（master节点维护）
        self.registered_nodes: Dict[str, NodeInfo] = {}
        
        # 监控数据
        self.local_memory_history: List[Dict[int, Dict[str, Any]]] = []
        self.global_memory_states: List[DistributedMemoryState] = []
        self.oom_events: List[DistributedOOMEvent] = []
        
        # 同步控制
        self.global_step = 0
        self.last_sync_time = 0
        self.sync_lock = threading.Lock()
        
        # 监控状态
        self.is_monitoring = False
        self.monitor_thread = None
        self.heartbeat_thread = None
        self.server_thread = None
        
        # 消息队列
        self.message_queue = queue.Queue()
        
        # 统计信息
        self.stats = {
            'total_sync_operations': 0,
            'failed_sync_operations': 0,
            'average_sync_time': 0.0,
            'node_failures': 0
        }
    
    def _generate_node_id(self) -> str:
        """生成节点ID"""
        return f"{self.hostname}_{int(time.time())}"
    
    def _get_local_ip(self) -> str:
        """获取本地IP地址"""
        try:
            # 连接到一个外部地址来获取本地IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"
    
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
    
    def _get_memory_info(self, device_id: int) -> Dict[str, Any]:
        """获取指定设备的显存信息"""
        try:
            result = subprocess.run([
                'nvidia-smi', f'--query-gpu=memory.used,memory.free,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits', f'--id={device_id}'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                used, free, total, utilization = map(int, result.stdout.strip().split(', '))
                return {
                    'used': used * 1024 * 1024,
                    'free': free * 1024 * 1024,
                    'total': total * 1024 * 1024,
                    'utilization': utilization,
                    'usage_ratio': used / total if total > 0 else 0.0
                }
        except Exception as e:
            print(f"Warning: Failed to get memory info for device {device_id}: {e}")
        
        return {'used': 0, 'free': 0, 'total': 0, 'utilization': 0, 'usage_ratio': 0.0}
    
    def _collect_local_memory_stats(self) -> Dict[int, Dict[str, Any]]:
        """收集本地内存统计信息"""
        local_stats = {}
        for device_id in self.devices:
            memory_info = self._get_memory_info(device_id)
            memory_info['node_id'] = self.node_id
            memory_info['timestamp'] = time.time()
            local_stats[device_id] = memory_info
        return local_stats
    
    def _start_server(self):
        """启动服务器（master节点）"""
        if self.role != DistributedRole.MASTER:
            return
        
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('', self.master_port))
            self.server_socket.listen(10)
            
            print(f"Master server started on {self.ip_address}:{self.master_port}")
            
            while self.is_monitoring:
                try:
                    client_socket, address = self.server_socket.accept()
                    # 在新线程中处理客户端连接
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, address),
                        daemon=True
                    )
                    client_thread.start()
                except Exception as e:
                    if self.is_monitoring:  # 只有在监控时才打印错误
                        print(f"Error accepting client connection: {e}")
        
        except Exception as e:
            print(f"Error starting server: {e}")
    
    def _handle_client(self, client_socket: socket.socket, address: Tuple[str, int]):
        """处理客户端连接"""
        try:
            while self.is_monitoring:
                data = client_socket.recv(4096)
                if not data:
                    break
                
                message = pickle.loads(data)
                self._process_message(message, client_socket)
                
        except Exception as e:
            print(f"Error handling client {address}: {e}")
        finally:
            client_socket.close()
    
    def _process_message(self, message: Dict[str, Any], client_socket: socket.socket):
        """处理接收到的消息"""
        msg_type = message.get('type')
        
        if msg_type == 'register':
            # 节点注册
            node_info = NodeInfo(
                node_id=message['node_id'],
                hostname=message['hostname'],
                ip_address=message['ip_address'],
                port=message.get('port', 0),
                role=DistributedRole(message['role']),
                gpu_devices=message['gpu_devices'],
                last_heartbeat=time.time()
            )
            self.registered_nodes[message['node_id']] = node_info
            
            # 发送确认
            response = {'type': 'register_ack', 'status': 'success'}
            client_socket.send(pickle.dumps(response))
            
        elif msg_type == 'memory_update':
            # 内存状态更新
            self._update_global_memory_state(message)
            
        elif msg_type == 'heartbeat':
            # 心跳消息
            node_id = message['node_id']
            if node_id in self.registered_nodes:
                self.registered_nodes[node_id].last_heartbeat = time.time()
    
    def _connect_to_master(self) -> bool:
        """连接到master节点"""
        if self.role == DistributedRole.MASTER or not self.master_address:
            return True
        
        try:
            master_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            master_socket.connect((self.master_address, self.master_port))
            
            # 发送注册消息
            register_msg = {
                'type': 'register',
                'node_id': self.node_id,
                'hostname': self.hostname,
                'ip_address': self.ip_address,
                'role': self.role.value,
                'gpu_devices': self.devices
            }
            
            master_socket.send(pickle.dumps(register_msg))
            
            # 等待确认
            response_data = master_socket.recv(1024)
            response = pickle.loads(response_data)
            
            if response.get('status') == 'success':
                self.client_connections['master'] = master_socket
                print(f"Successfully registered with master at {self.master_address}:{self.master_port}")
                return True
            else:
                print(f"Failed to register with master: {response}")
                return False
                
        except Exception as e:
            print(f"Error connecting to master: {e}")
            return False
    
    def _send_memory_update(self, memory_stats: Dict[int, Dict[str, Any]]):
        """发送内存更新到master节点"""
        if self.role == DistributedRole.MASTER or 'master' not in self.client_connections:
            return
        
        try:
            update_msg = {
                'type': 'memory_update',
                'node_id': self.node_id,
                'global_step': self.global_step,
                'memory_stats': memory_stats,
                'timestamp': time.time()
            }
            
            master_socket = self.client_connections['master']
            master_socket.send(pickle.dumps(update_msg))
            
        except Exception as e:
            print(f"Error sending memory update: {e}")
    
    def _send_heartbeat(self):
        """发送心跳消息"""
        if self.role == DistributedRole.MASTER or 'master' not in self.client_connections:
            return
        
        try:
            heartbeat_msg = {
                'type': 'heartbeat',
                'node_id': self.node_id,
                'timestamp': time.time()
            }
            
            master_socket = self.client_connections['master']
            master_socket.send(pickle.dumps(heartbeat_msg))
            
        except Exception as e:
            print(f"Error sending heartbeat: {e}")
    
    def _heartbeat_loop(self):
        """心跳循环"""
        while self.is_monitoring:
            self._send_heartbeat()
            time.sleep(self.heartbeat_interval)
    
    def _update_global_memory_state(self, message: Dict[str, Any]):
        """更新全局内存状态（master节点）"""
        if self.role != DistributedRole.MASTER:
            return
        
        node_id = message['node_id']
        memory_stats = message['memory_stats']
        
        # 构建全局状态
        global_state = self._build_global_memory_state()
        self.global_memory_states.append(global_state)
        
        # 维护历史记录大小
        max_history = 1000
        if len(self.global_memory_states) > max_history:
            self.global_memory_states = self.global_memory_states[-max_history:]
    
    def _build_global_memory_state(self) -> DistributedMemoryState:
        """构建全局内存状态"""
        current_time = time.time()
        
        # 收集所有节点的内存信息
        all_nodes_memory = {}
        total_memory_usage = 0
        memory_ratios = []
        
        for node_id, node_info in self.registered_nodes.items():
            if not node_info.is_active:
                continue
            
            # 这里简化实现，实际应该从各节点获取最新数据
            node_memory = {}
            for device_id in node_info.gpu_devices:
                memory_info = self._get_memory_info(device_id)
                node_memory[device_id] = memory_info
                total_memory_usage += memory_info['used']
                memory_ratios.append(memory_info['usage_ratio'])
            
            all_nodes_memory[node_id] = node_memory
        
        # 计算内存平衡比率
        if memory_ratios:
            memory_balance_ratio = 1.0 - (np.std(memory_ratios) / np.mean(memory_ratios)) if np.mean(memory_ratios) > 0 else 1.0
        else:
            memory_balance_ratio = 1.0
        
        # 检测瓶颈节点
        bottleneck_nodes = []
        if memory_ratios:
            avg_ratio = np.mean(memory_ratios)
            for node_id, node_memory in all_nodes_memory.items():
                node_avg_ratio = np.mean([info['usage_ratio'] for info in node_memory.values()])
                if node_avg_ratio > avg_ratio * 1.2:  # 超过平均值20%
                    bottleneck_nodes.append(node_id)
        
        # 确定同步状态
        sync_status = self._determine_sync_status()
        
        # 生成建议
        recommendations = self._generate_recommendations(memory_balance_ratio, bottleneck_nodes)
        
        return DistributedMemoryState(
            timestamp=current_time,
            global_step=self.global_step,
            nodes=all_nodes_memory,
            memory_balance_ratio=memory_balance_ratio,
            sync_status=sync_status,
            bottleneck_nodes=bottleneck_nodes,
            total_memory_usage=total_memory_usage,
            recommendations=recommendations
        )
    
    def _determine_sync_status(self) -> SyncStatus:
        """确定同步状态"""
        current_time = time.time()
        
        # 检查所有节点的最后心跳时间
        for node_info in self.registered_nodes.values():
            if current_time - node_info.last_heartbeat > self.sync_timeout:
                return SyncStatus.COMMUNICATION_ERROR
        
        # 简化的同步检查逻辑
        if len(self.registered_nodes) > 1:
            return SyncStatus.IN_SYNC  # 假设同步正常
        else:
            return SyncStatus.UNKNOWN
    
    def _generate_recommendations(self, balance_ratio: float, bottleneck_nodes: List[str]) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        if balance_ratio < 0.8:
            recommendations.append("Memory usage is unbalanced across nodes")
            recommendations.append("Consider redistributing workload or adjusting batch sizes")
        
        if bottleneck_nodes:
            recommendations.append(f"Bottleneck nodes detected: {', '.join(bottleneck_nodes)}")
            recommendations.append("Consider reducing workload on bottleneck nodes")
        
        if not recommendations:
            recommendations.append("Memory usage is well balanced across the cluster")
        
        return recommendations
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                # 收集本地内存统计
                local_stats = self._collect_local_memory_stats()
                self.local_memory_history.append(local_stats)
                
                # 维护历史记录大小
                max_local_history = 500
                if len(self.local_memory_history) > max_local_history:
                    self.local_memory_history = self.local_memory_history[-max_local_history:]
                
                # 发送给master节点（如果是worker）
                if self.role == DistributedRole.WORKER:
                    self._send_memory_update(local_stats)
                
                # 如果是master，更新全局状态
                elif self.role == DistributedRole.MASTER:
                    global_state = self._build_global_memory_state()
                    self.global_memory_states.append(global_state)
                
                # 检测OOM风险
                self._check_oom_risk(local_stats)
                
                self.global_step += 1
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                print(f"Error in distributed monitoring loop: {e}")
                time.sleep(5)
    
    def _check_oom_risk(self, local_stats: Dict[int, Dict[str, Any]]):
        """检查OOM风险"""
        for device_id, memory_info in local_stats.items():
            if memory_info['usage_ratio'] > 0.9:  # 90%使用率
                print(f"⚠️  High memory usage on GPU {device_id}: {memory_info['usage_ratio']:.1%}")
                
                # 记录OOM风险事件
                if memory_info['usage_ratio'] > 0.95:  # 95%使用率，OOM风险很高
                    oom_event = DistributedOOMEvent(
                        timestamp=time.time(),
                        node_id=self.node_id,
                        device_id=device_id,
                        global_step=self.global_step,
                        oom_type='single_gpu',
                        affected_nodes=[self.node_id],
                        recovery_actions=['Reduce batch size', 'Clear cache', 'Restart training']
                    )
                    self.oom_events.append(oom_event)
    
    def start_monitoring(self):
        """开始监控"""
        if self.is_monitoring:
            print("Distributed memory monitor is already running")
            return
        
        self.is_monitoring = True
        
        # 启动服务器（master节点）
        if self.role == DistributedRole.MASTER:
            self.server_thread = threading.Thread(target=self._start_server, daemon=True)
            self.server_thread.start()
            time.sleep(1)  # 等待服务器启动
        
        # 连接到master（worker节点）
        elif self.role == DistributedRole.WORKER:
            if not self._connect_to_master():
                print("Failed to connect to master, running in standalone mode")
                self.role = DistributedRole.STANDALONE
        
        # 启动监控循环
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        # 启动心跳循环（worker节点）
        if self.role == DistributedRole.WORKER:
            self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self.heartbeat_thread.start()
        
        print(f"Started distributed memory monitoring (Role: {self.role.value})")
        print(f"Node ID: {self.node_id}, Monitoring devices: {self.devices}")
    
    def stop_monitoring(self):
        """停止监控"""
        if not self.is_monitoring:
            print("Distributed memory monitor is not running")
            return
        
        self.is_monitoring = False
        
        # 关闭服务器socket
        if self.server_socket:
            self.server_socket.close()
        
        # 关闭客户端连接
        for conn in self.client_connections.values():
            conn.close()
        
        # 等待线程结束
        for thread in [self.monitor_thread, self.heartbeat_thread, self.server_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=3.0)
        
        print("Stopped distributed memory monitoring")
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """获取集群状态"""
        if self.role != DistributedRole.MASTER:
            return {'error': 'Only master node can provide cluster status'}
        
        current_time = time.time()
        active_nodes = []
        inactive_nodes = []
        
        for node_id, node_info in self.registered_nodes.items():
            if current_time - node_info.last_heartbeat <= self.sync_timeout:
                active_nodes.append(node_id)
                node_info.is_active = True
            else:
                inactive_nodes.append(node_id)
                node_info.is_active = False
        
        # 获取最新的全局状态
        latest_state = self.global_memory_states[-1] if self.global_memory_states else None
        
        return {
            'cluster_id': f"cluster_{self.node_id}",
            'master_node': self.node_id,
            'total_nodes': len(self.registered_nodes),
            'active_nodes': len(active_nodes),
            'inactive_nodes': len(inactive_nodes),
            'active_node_list': active_nodes,
            'inactive_node_list': inactive_nodes,
            'total_gpus': sum(len(node.gpu_devices) for node in self.registered_nodes.values()),
            'last_sync_time': self.last_sync_time,
            'global_step': self.global_step,
            'memory_balance_ratio': latest_state.memory_balance_ratio if latest_state else 0.0,
            'sync_status': latest_state.sync_status.value if latest_state else 'unknown',
            'bottleneck_nodes': latest_state.bottleneck_nodes if latest_state else [],
            'total_memory_usage_gb': (latest_state.total_memory_usage / (1024**3)) if latest_state else 0.0
        }
    
    def get_memory_distribution(self) -> Dict[str, Any]:
        """获取内存分布信息"""
        if not self.local_memory_history:
            return {'error': 'No memory data available'}
        
        latest_stats = self.local_memory_history[-1]
        
        distribution = {
            'node_id': self.node_id,
            'timestamp': time.time(),
            'devices': {},
            'node_summary': {
                'total_memory_gb': 0,
                'used_memory_gb': 0,
                'free_memory_gb': 0,
                'average_utilization': 0.0
            }
        }
        
        total_memory = 0
        used_memory = 0
        utilizations = []
        
        for device_id, memory_info in latest_stats.items():
            device_info = {
                'total_gb': memory_info['total'] / (1024**3),
                'used_gb': memory_info['used'] / (1024**3),
                'free_gb': memory_info['free'] / (1024**3),
                'utilization_percent': memory_info['utilization'],
                'usage_ratio': memory_info['usage_ratio']
            }
            distribution['devices'][device_id] = device_info
            
            total_memory += memory_info['total']
            used_memory += memory_info['used']
            utilizations.append(memory_info['utilization'])
        
        distribution['node_summary'] = {
            'total_memory_gb': total_memory / (1024**3),
            'used_memory_gb': used_memory / (1024**3),
            'free_memory_gb': (total_memory - used_memory) / (1024**3),
            'average_utilization': np.mean(utilizations) if utilizations else 0.0
        }
        
        return distribution
    
    def generate_distributed_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """生成分布式监控报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'report_type': 'distributed_memory_monitoring',
            'cluster_info': {},
            'node_details': {},
            'memory_analysis': {},
            'oom_events': [],
            'recommendations': []
        }
        
        # 集群信息
        if self.role == DistributedRole.MASTER:
            report['cluster_info'] = self.get_cluster_status()
        else:
            report['cluster_info'] = {
                'role': self.role.value,
                'node_id': self.node_id,
                'master_address': self.master_address
            }
        
        # 节点详细信息
        report['node_details'] = self.get_memory_distribution()
        
        # 内存分析
        if self.global_memory_states:
            latest_global = self.global_memory_states[-1]
            report['memory_analysis'] = {
                'memory_balance_ratio': latest_global.memory_balance_ratio,
                'sync_status': latest_global.sync_status.value,
                'bottleneck_nodes': latest_global.bottleneck_nodes,
                'total_memory_usage_gb': latest_global.total_memory_usage / (1024**3),
                'global_step': latest_global.global_step
            }
        
        # OOM事件
        for oom_event in self.oom_events[-10:]:  # 最近10个事件
            report['oom_events'].append({
                'timestamp': datetime.fromtimestamp(oom_event.timestamp).isoformat(),
                'node_id': oom_event.node_id,
                'device_id': oom_event.device_id,
                'oom_type': oom_event.oom_type,
                'affected_nodes': oom_event.affected_nodes,
                'recovery_actions': oom_event.recovery_actions
            })
        
        # 生成建议
        if self.global_memory_states:
            latest_recommendations = self.global_memory_states[-1].recommendations
            report['recommendations'] = latest_recommendations
        else:
            report['recommendations'] = ["Enable distributed monitoring for comprehensive recommendations"]
        
        # 保存报告
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Distributed memory report saved to {output_file}")
        
        return report


if __name__ == "__main__":
    import sys
    
    # 演示用法
    if len(sys.argv) > 1 and sys.argv[1] == "master":
        # 启动master节点
        monitor = DistributedMemoryMonitor(
            role=DistributedRole.MASTER,
            sampling_interval=3.0
        )
    else:
        # 启动worker节点（需要指定master地址）
        monitor = DistributedMemoryMonitor(
            role=DistributedRole.WORKER,
            master_address="localhost",  # 实际使用时应该是master节点的IP
            sampling_interval=3.0
        )
    
    try:
        monitor.start_monitoring()
        
        # 运行60秒
        time.sleep(60)
        
        # 获取状态信息
        if monitor.role == DistributedRole.MASTER:
            cluster_status = monitor.get_cluster_status()
            print("\n=== Cluster Status ===")
            print(f"Total Nodes: {cluster_status['total_nodes']}")
            print(f"Active Nodes: {cluster_status['active_nodes']}")
            print(f"Total GPUs: {cluster_status['total_gpus']}")
            print(f"Memory Balance Ratio: {cluster_status['memory_balance_ratio']:.2f}")
        
        # 获取内存分布
        memory_dist = monitor.get_memory_distribution()
        if 'error' not in memory_dist:
            print(f"\n=== Node {memory_dist['node_id']} Memory Distribution ===")
            summary = memory_dist['node_summary']
            print(f"Total Memory: {summary['total_memory_gb']:.2f} GB")
            print(f"Used Memory: {summary['used_memory_gb']:.2f} GB")
            print(f"Average Utilization: {summary['average_utilization']:.1f}%")
        
        # 生成报告
        report = monitor.generate_distributed_report("distributed_memory_report.json")
        print(f"\nGenerated distributed memory monitoring report")
        
    finally:
        monitor.stop_monitoring()
