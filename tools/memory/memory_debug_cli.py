#!/usr/bin/env python3
"""
Memory Debug CLI - 显存调试工具命令行界面
统一的命令行工具来使用各种显存调试功能

使用方法:
    python memory_debug_cli.py profile --duration 60 --output profile_report.json
    python memory_debug_cli.py oom --threshold 85 --monitor
    python memory_debug_cli.py leak --enable-tracking --output leak_report.json
    python memory_debug_cli.py fragment --analyze --plot
    python memory_debug_cli.py distributed --role master --port 29500
"""

import argparse
import sys
import time
import json
import signal
import threading
from typing import Optional, List, Dict, Any

# 导入各个工具模块
from memory_profiler import MemoryProfiler
from oom_detector import OOMDetector, default_warning_callback
from memory_leak_detector import MemoryLeakDetector
from fragmentation_analyzer import FragmentationAnalyzer
from distributed_memory_monitor import DistributedMemoryMonitor, DistributedRole


class MemoryDebugCLI:
    """显存调试工具命令行界面"""
    
    def __init__(self):
        self.active_monitors = []
        self.stop_event = threading.Event()
        
        # 注册信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """信号处理器 - 优雅退出"""
        print("\n🔄 Gracefully shutting down memory debugging tools...")
        self.stop_event.set()
        
        for monitor in self.active_monitors:
            if hasattr(monitor, 'stop_monitoring'):
                monitor.stop_monitoring()
        
        print("✅ All monitors stopped successfully")
        sys.exit(0)
    
    def profile_command(self, args):
        """内存分析命令"""
        print("🔍 Starting Memory Profiler...")
        print(f"   Duration: {args.duration} seconds")
        print(f"   Devices: {args.devices or 'All available'}")
        print(f"   Sampling interval: {args.interval} seconds")
        
        profiler = MemoryProfiler(
            devices=args.devices,
            sampling_interval=args.interval,
            max_snapshots=args.max_snapshots
        )
        self.active_monitors.append(profiler)
        
        try:
            profiler.start_monitoring()
            
            # 监控指定时间
            if args.duration > 0:
                print(f"📊 Monitoring for {args.duration} seconds...")
                for i in range(args.duration):
                    if self.stop_event.is_set():
                        break
                    time.sleep(1)
                    if (i + 1) % 10 == 0:
                        print(f"   Progress: {i + 1}/{args.duration} seconds")
            else:
                print("📊 Monitoring indefinitely (Press Ctrl+C to stop)...")
                self.stop_event.wait()
            
            profiler.stop_monitoring()
            
            # 生成报告
            if args.output:
                report = profiler.generate_report(args.output)
                print(f"📄 Report saved to: {args.output}")
            
            # 显示当前状态
            status = profiler.get_current_status()
            print("\n📈 Current Memory Status:")
            for device_id, info in status.items():
                print(f"   GPU {device_id}: {info['utilization']:.1f}% used "
                      f"({info['current_used']/(1024**3):.2f} GB / "
                      f"{info['total']/(1024**3):.2f} GB)")
            
            # 生成图表
            if args.plot and status:
                for device_id in status.keys():
                    plot_file = f"gpu_{device_id}_memory_usage.png"
                    profiler.plot_memory_usage(device_id, plot_file)
                    print(f"📊 Memory usage plot saved to: {plot_file}")
        
        except Exception as e:
            print(f"❌ Error in memory profiling: {e}")
        finally:
            if profiler in self.active_monitors:
                self.active_monitors.remove(profiler)
    
    def oom_command(self, args):
        """OOM检测命令"""
        print("🚨 Starting OOM Detector...")
        print(f"   Warning threshold: {args.threshold}%")
        print(f"   Critical threshold: {args.critical}%")
        print(f"   Prediction window: {args.window} snapshots")
        
        detector = OOMDetector(
            devices=args.devices,
            warning_threshold=args.threshold / 100.0,
            critical_threshold=args.critical / 100.0,
            prediction_window=args.window,
            sampling_interval=args.interval
        )
        self.active_monitors.append(detector)
        
        # 添加默认回调
        detector.add_warning_callback(default_warning_callback)
        
        try:
            detector.start_monitoring()
            
            if args.monitor:
                print("🔍 Monitoring for OOM risks (Press Ctrl+C to stop)...")
                self.stop_event.wait()
            else:
                print(f"🔍 Monitoring for {args.duration} seconds...")
                for i in range(args.duration):
                    if self.stop_event.is_set():
                        break
                    time.sleep(1)
            
            detector.stop_monitoring()
            
            # 获取风险评估
            risks = detector.get_risk_assessment()
            print("\n⚠️  OOM Risk Assessment:")
            for device_id, prediction in risks.items():
                risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴", "critical": "💥"}
                emoji = risk_emoji.get(prediction.risk_level.value, "❓")
                print(f"   {emoji} GPU {device_id}: {prediction.risk_level.value.upper()} risk")
                print(f"      Current usage: {prediction.current_usage*100:.1f}%")
                print(f"      Growth rate: {prediction.growth_rate:.2f} MB/s")
                if prediction.predicted_oom_time:
                    print(f"      Predicted OOM in: {prediction.predicted_oom_time:.0f} seconds")
                print(f"      Recommendation: {prediction.recommendation}")
            
            # 生成报告
            if args.output:
                report = detector.generate_oom_report(args.output)
                print(f"📄 OOM risk report saved to: {args.output}")
        
        except Exception as e:
            print(f"❌ Error in OOM detection: {e}")
        finally:
            if detector in self.active_monitors:
                self.active_monitors.remove(detector)
    
    def leak_command(self, args):
        """内存泄漏检测命令"""
        print("🔍 Starting Memory Leak Detector...")
        print(f"   Leak threshold: {args.threshold} MB/min")
        print(f"   Reference tracking: {'Enabled' if args.enable_tracking else 'Disabled'}")
        print(f"   History window: {args.window} snapshots")
        
        detector = MemoryLeakDetector(
            devices=args.devices,
            sampling_interval=args.interval,
            leak_threshold=args.threshold * 1024 * 1024,  # 转换为bytes/min
            history_window=args.window,
            enable_reference_tracking=args.enable_tracking
        )
        self.active_monitors.append(detector)
        
        try:
            detector.start_monitoring()
            
            print(f"🔍 Monitoring for memory leaks for {args.duration} seconds...")
            for i in range(args.duration):
                if self.stop_event.is_set():
                    break
                time.sleep(1)
                if (i + 1) % 30 == 0:
                    print(f"   Progress: {i + 1}/{args.duration} seconds")
            
            detector.stop_monitoring()
            
            # 获取泄漏摘要
            summary = detector.get_leak_summary()
            if 'error' not in summary:
                print("\n🔍 Memory Leak Detection Summary:")
                print(f"   Monitoring duration: {summary['monitoring_duration_hours']:.2f} hours")
                print(f"   Total leaks detected: {summary['total_leaks_detected']}")
                print(f"   Total leaked memory: {summary['total_leaked_memory_mb']:.2f} MB")
                print(f"   Leak rate: {summary['leak_rate_per_hour']:.1f} leaks/hour")
                
                if summary['severity_distribution']:
                    print("   Severity distribution:")
                    for severity, count in summary['severity_distribution'].items():
                        print(f"      {severity}: {count}")
            
            # 获取修复建议
            if args.devices or detector.devices:
                test_devices = args.devices or detector.devices[:1]
                for device_id in test_devices:
                    suggestions = detector.suggest_fixes(device_id)
                    if suggestions:
                        print(f"\n💡 Fix Suggestions for GPU {device_id}:")
                        for suggestion in suggestions[:5]:  # 显示前5个建议
                            print(f"   • {suggestion}")
            
            # 生成报告
            if args.output:
                report = detector.generate_leak_report(args.output)
                print(f"📄 Memory leak report saved to: {args.output}")
        
        except Exception as e:
            print(f"❌ Error in memory leak detection: {e}")
        finally:
            if detector in self.active_monitors:
                self.active_monitors.remove(detector)
    
    def fragment_command(self, args):
        """内存碎片分析命令"""
        print("🧩 Starting Fragmentation Analyzer...")
        print(f"   Devices: {args.devices or 'All available'}")
        print(f"   Analysis duration: {args.duration} seconds")
        
        analyzer = FragmentationAnalyzer(
            devices=args.devices,
            sampling_interval=args.interval,
            max_history=args.max_history
        )
        self.active_monitors.append(analyzer)
        
        try:
            analyzer.start_monitoring()
            
            print(f"🔍 Analyzing memory fragmentation for {args.duration} seconds...")
            for i in range(args.duration):
                if self.stop_event.is_set():
                    break
                time.sleep(1)
                if (i + 1) % 15 == 0:
                    print(f"   Progress: {i + 1}/{args.duration} seconds")
            
            analyzer.stop_monitoring()
            
            # 获取当前碎片化状态
            current_frag = analyzer.get_current_fragmentation()
            print("\n🧩 Current Fragmentation Status:")
            for device_id, metrics in current_frag.items():
                frag_emoji = {
                    "low": "🟢", "moderate": "🟡", "high": "🔴", "severe": "💥"
                }
                emoji = frag_emoji.get(metrics.fragmentation_level.value, "❓")
                print(f"   {emoji} GPU {device_id}: {metrics.fragmentation_level.value.upper()}")
                print(f"      Fragmentation ratio: {metrics.fragmentation_ratio:.2%}")
                print(f"      Free blocks: {metrics.num_free_blocks}")
                print(f"      Largest free block: {metrics.largest_free_block/(1024**2):.1f} MB")
                print(f"      Allocation efficiency: {metrics.allocation_efficiency:.2%}")
            
            # 生成碎片整理建议
            if current_frag:
                print("\n💡 Defragmentation Suggestions:")
                for device_id in current_frag.keys():
                    suggestions = analyzer.suggest_defragmentation(device_id)
                    if suggestions['recommendations']:
                        print(f"   GPU {device_id} (Priority: {suggestions['priority']}):")
                        for rec in suggestions['recommendations'][:3]:
                            print(f"      • {rec}")
            
            # 生成图表
            if args.plot and current_frag:
                for device_id in current_frag.keys():
                    plot_file = f"gpu_{device_id}_fragmentation.png"
                    analyzer.plot_fragmentation_timeline(device_id, plot_file)
                    print(f"📊 Fragmentation plot saved to: {plot_file}")
            
            # 生成报告
            if args.output:
                report = analyzer.generate_fragmentation_report(args.output)
                print(f"📄 Fragmentation report saved to: {args.output}")
        
        except Exception as e:
            print(f"❌ Error in fragmentation analysis: {e}")
        finally:
            if analyzer in self.active_monitors:
                self.active_monitors.remove(analyzer)
    
    def distributed_command(self, args):
        """分布式监控命令"""
        print("🌐 Starting Distributed Memory Monitor...")
        print(f"   Role: {args.role}")
        if args.role == 'master':
            print(f"   Port: {args.port}")
        else:
            print(f"   Master address: {args.master_host}:{args.port}")
        
        role = DistributedRole.MASTER if args.role == 'master' else DistributedRole.WORKER
        
        monitor = DistributedMemoryMonitor(
            role=role,
            master_address=args.master_host if args.role != 'master' else None,
            master_port=args.port,
            devices=args.devices,
            sampling_interval=args.interval
        )
        self.active_monitors.append(monitor)
        
        try:
            monitor.start_monitoring()
            
            print(f"🔍 Running distributed monitoring for {args.duration} seconds...")
            for i in range(args.duration):
                if self.stop_event.is_set():
                    break
                time.sleep(1)
                if (i + 1) % 30 == 0:
                    print(f"   Progress: {i + 1}/{args.duration} seconds")
                    
                    # 显示状态更新
                    if role == DistributedRole.MASTER:
                        status = monitor.get_cluster_status()
                        if 'error' not in status:
                            print(f"   Cluster: {status['active_nodes']}/{status['total_nodes']} nodes active")
            
            monitor.stop_monitoring()
            
            # 显示集群状态
            if role == DistributedRole.MASTER:
                status = monitor.get_cluster_status()
                if 'error' not in status:
                    print("\n🌐 Final Cluster Status:")
                    print(f"   Total nodes: {status['total_nodes']}")
                    print(f"   Active nodes: {status['active_nodes']}")
                    print(f"   Total GPUs: {status['total_gpus']}")
                    print(f"   Memory balance ratio: {status['memory_balance_ratio']:.2f}")
                    print(f"   Sync status: {status['sync_status']}")
                    if status['bottleneck_nodes']:
                        print(f"   Bottleneck nodes: {', '.join(status['bottleneck_nodes'])}")
            
            # 显示内存分布
            memory_dist = monitor.get_memory_distribution()
            if 'error' not in memory_dist:
                print(f"\n💾 Node {memory_dist['node_id']} Memory Distribution:")
                summary = memory_dist['node_summary']
                print(f"   Total memory: {summary['total_memory_gb']:.2f} GB")
                print(f"   Used memory: {summary['used_memory_gb']:.2f} GB")
                print(f"   Average utilization: {summary['average_utilization']:.1f}%")
                
                for device_id, device_info in memory_dist['devices'].items():
                    print(f"   GPU {device_id}: {device_info['usage_ratio']:.1%} "
                          f"({device_info['used_gb']:.2f} GB / {device_info['total_gb']:.2f} GB)")
            
            # 生成报告
            if args.output:
                report = monitor.generate_distributed_report(args.output)
                print(f"📄 Distributed monitoring report saved to: {args.output}")
        
        except Exception as e:
            print(f"❌ Error in distributed monitoring: {e}")
        finally:
            if monitor in self.active_monitors:
                self.active_monitors.remove(monitor)
    
    def interactive_command(self, args):
        """交互式模式"""
        print("🎮 Entering Interactive Memory Debugging Mode")
        print("Available commands: profile, oom, leak, fragment, distributed, exit")
        
        while True:
            try:
                command = input("\nmemory-debug> ").strip().split()
                if not command:
                    continue
                
                if command[0] == 'exit':
                    break
                elif command[0] == 'profile':
                    # 简化的交互式profiling
                    duration = int(input("Duration (seconds): ") or "30")
                    self.profile_command(type('Args', (), {
                        'duration': duration, 'devices': None, 'interval': 1.0,
                        'max_snapshots': 1000, 'output': 'interactive_profile.json', 'plot': True
                    })())
                elif command[0] == 'oom':
                    # 简化的交互式OOM检测
                    threshold = float(input("Warning threshold (0-100): ") or "85")
                    self.oom_command(type('Args', (), {
                        'threshold': threshold, 'critical': 95, 'devices': None,
                        'window': 30, 'interval': 2.0, 'duration': 60,
                        'monitor': False, 'output': 'interactive_oom.json'
                    })())
                elif command[0] == 'leak':
                    # 简化的交互式泄漏检测
                    threshold = float(input("Leak threshold (MB/min): ") or "10")
                    self.leak_command(type('Args', (), {
                        'threshold': threshold, 'devices': None, 'interval': 5.0,
                        'window': 100, 'duration': 120, 'enable_tracking': True,
                        'output': 'interactive_leak.json'
                    })())
                elif command[0] == 'fragment':
                    # 简化的交互式碎片分析
                    self.fragment_command(type('Args', (), {
                        'devices': None, 'interval': 2.0, 'duration': 60,
                        'max_history': 500, 'plot': True, 'output': 'interactive_fragment.json'
                    })())
                else:
                    print("Unknown command. Available: profile, oom, leak, fragment, exit")
            
            except KeyboardInterrupt:
                print("\nExiting interactive mode...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def run(self):
        """运行CLI"""
        parser = argparse.ArgumentParser(
            description="Memory Debug CLI - 显存调试工具集",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # 内存分析 - 监控60秒并生成报告
  python memory_debug_cli.py profile --duration 60 --output profile.json --plot
  
  # OOM检测 - 85%阈值持续监控
  python memory_debug_cli.py oom --threshold 85 --monitor
  
  # 泄漏检测 - 启用引用跟踪，10MB/min阈值
  python memory_debug_cli.py leak --threshold 10 --enable-tracking --duration 300
  
  # 碎片分析 - 分析并生成图表
  python memory_debug_cli.py fragment --duration 120 --plot --output fragment.json
  
  # 分布式监控 - 启动master节点
  python memory_debug_cli.py distributed --role master --port 29500
  
  # 分布式监控 - 启动worker节点
  python memory_debug_cli.py distributed --role worker --master-host 192.168.1.100
  
  # 交互式模式
  python memory_debug_cli.py interactive
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Profile命令
        profile_parser = subparsers.add_parser('profile', help='Memory profiling')
        profile_parser.add_argument('--duration', type=int, default=60,
                                   help='Monitoring duration in seconds (default: 60)')
        profile_parser.add_argument('--devices', type=int, nargs='+',
                                   help='GPU device IDs to monitor (default: all)')
        profile_parser.add_argument('--interval', type=float, default=1.0,
                                   help='Sampling interval in seconds (default: 1.0)')
        profile_parser.add_argument('--max-snapshots', type=int, default=10000,
                                   help='Maximum snapshots to keep (default: 10000)')
        profile_parser.add_argument('--output', type=str,
                                   help='Output report file (JSON format)')
        profile_parser.add_argument('--plot', action='store_true',
                                   help='Generate memory usage plots')
        
        # OOM命令
        oom_parser = subparsers.add_parser('oom', help='OOM detection')
        oom_parser.add_argument('--threshold', type=float, default=85,
                               help='Warning threshold percentage (default: 85)')
        oom_parser.add_argument('--critical', type=float, default=95,
                               help='Critical threshold percentage (default: 95)')
        oom_parser.add_argument('--devices', type=int, nargs='+',
                               help='GPU device IDs to monitor (default: all)')
        oom_parser.add_argument('--window', type=int, default=30,
                               help='Prediction window size (default: 30)')
        oom_parser.add_argument('--interval', type=float, default=2.0,
                               help='Sampling interval in seconds (default: 2.0)')
        oom_parser.add_argument('--duration', type=int, default=300,
                               help='Monitoring duration in seconds (default: 300)')
        oom_parser.add_argument('--monitor', action='store_true',
                               help='Continuous monitoring mode')
        oom_parser.add_argument('--output', type=str,
                               help='Output report file (JSON format)')
        
        # Leak命令
        leak_parser = subparsers.add_parser('leak', help='Memory leak detection')
        leak_parser.add_argument('--threshold', type=float, default=10,
                                help='Leak threshold in MB/min (default: 10)')
        leak_parser.add_argument('--devices', type=int, nargs='+',
                                help='GPU device IDs to monitor (default: all)')
        leak_parser.add_argument('--interval', type=float, default=5.0,
                                help='Sampling interval in seconds (default: 5.0)')
        leak_parser.add_argument('--window', type=int, default=100,
                                help='History window size (default: 100)')
        leak_parser.add_argument('--duration', type=int, default=600,
                                help='Monitoring duration in seconds (default: 600)')
        leak_parser.add_argument('--enable-tracking', action='store_true',
                                help='Enable reference tracking')
        leak_parser.add_argument('--output', type=str,
                                help='Output report file (JSON format)')
        
        # Fragment命令
        fragment_parser = subparsers.add_parser('fragment', help='Memory fragmentation analysis')
        fragment_parser.add_argument('--devices', type=int, nargs='+',
                                    help='GPU device IDs to monitor (default: all)')
        fragment_parser.add_argument('--interval', type=float, default=2.0,
                                    help='Sampling interval in seconds (default: 2.0)')
        fragment_parser.add_argument('--duration', type=int, default=120,
                                    help='Analysis duration in seconds (default: 120)')
        fragment_parser.add_argument('--max-history', type=int, default=1000,
                                    help='Maximum history records (default: 1000)')
        fragment_parser.add_argument('--plot', action='store_true',
                                    help='Generate fragmentation plots')
        fragment_parser.add_argument('--output', type=str,
                                    help='Output report file (JSON format)')
        
        # Distributed命令
        distributed_parser = subparsers.add_parser('distributed', help='Distributed memory monitoring')
        distributed_parser.add_argument('--role', choices=['master', 'worker'], required=True,
                                       help='Node role: master or worker')
        distributed_parser.add_argument('--master-host', type=str, default='localhost',
                                       help='Master node hostname/IP (for worker nodes)')
        distributed_parser.add_argument('--port', type=int, default=29500,
                                       help='Communication port (default: 29500)')
        distributed_parser.add_argument('--devices', type=int, nargs='+',
                                       help='GPU device IDs to monitor (default: all)')
        distributed_parser.add_argument('--interval', type=float, default=3.0,
                                       help='Sampling interval in seconds (default: 3.0)')
        distributed_parser.add_argument('--duration', type=int, default=300,
                                       help='Monitoring duration in seconds (default: 300)')
        distributed_parser.add_argument('--output', type=str,
                                       help='Output report file (JSON format)')
        
        # Interactive命令
        interactive_parser = subparsers.add_parser('interactive', help='Interactive mode')
        
        args = parser.parse_args()
        
        if not args.command:
            parser.print_help()
            return
        
        print("🚀 Memory Debug CLI - Tiny-Torch Memory Debugging Tools")
        print("=" * 60)
        
        try:
            if args.command == 'profile':
                self.profile_command(args)
            elif args.command == 'oom':
                self.oom_command(args)
            elif args.command == 'leak':
                self.leak_command(args)
            elif args.command == 'fragment':
                self.fragment_command(args)
            elif args.command == 'distributed':
                self.distributed_command(args)
            elif args.command == 'interactive':
                self.interactive_command(args)
        
        except KeyboardInterrupt:
            print("\n🔄 Operation interrupted by user")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n✅ Memory debugging session completed")


if __name__ == "__main__":
    cli = MemoryDebugCLI()
    cli.run()
