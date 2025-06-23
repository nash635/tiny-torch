# Memory Debugging Tools for Distributed Training
# 分布式训练显存调试工具集

这是一套专为分布式训练场景设计的显存调试工具，帮助开发者排查和解决常见的显存问题。

## 🎯 主要功能

### 1. 显存分析工具 (Memory Profiler)
- **实时显存监控**: 持续监控GPU显存使用情况
- **峰值追踪**: 记录显存使用峰值，帮助优化内存配置
- **使用热图**: 生成显存使用趋势图表
- **详细报告**: 输出JSON格式的详细分析报告

### 2. OOM检测器 (OOM Detector)
- **智能预警**: 基于趋势分析预测OOM发生时间
- **风险评估**: 多级风险评估（低/中/高/严重）
- **自动建议**: 提供针对性的优化建议
- **实时监控**: 支持持续监控模式

### 3. 显存碎片分析器 (Fragmentation Analyzer)
- **碎片检测**: 分析显存碎片化程度
- **分配模式分析**: 识别内存分配模式
- **整理建议**: 提供碎片整理建议
- **趋势预测**: 预测碎片化发展趋势

### 4. 内存泄漏检测器 (Memory Leak Detector)
- **泄漏检测**: 检测显存和Python内存泄漏
- **来源分析**: 分析潜在的泄漏来源
- **引用跟踪**: 跟踪对象引用关系
- **修复建议**: 提供具体的修复建议

### 5. 分布式内存监控器 (Distributed Memory Monitor)
- **多节点协调**: 跨节点显存使用同步监控
- **负载均衡分析**: 分析各节点间的内存负载平衡
- **瓶颈检测**: 识别内存使用瓶颈节点
- **集群级OOM预防**: 集群级别的OOM预防和恢复

## 🚀 快速开始

### 安装依赖
```bash
pip install numpy matplotlib psutil
```

### 基本使用

#### 1. 内存分析
```bash
# 监控60秒并生成报告和图表
python memory_debug_cli.py profile --duration 60 --output profile.json --plot

# 指定特定GPU设备
python memory_debug_cli.py profile --devices 0 1 --duration 120
```

#### 2. OOM风险检测
```bash
# 85%阈值持续监控
python memory_debug_cli.py oom --threshold 85 --monitor

# 定时监控300秒
python memory_debug_cli.py oom --threshold 80 --duration 300 --output oom_report.json
```

#### 3. 内存泄漏检测
```bash
# 启用引用跟踪，10MB/min泄漏阈值
python memory_debug_cli.py leak --threshold 10 --enable-tracking --duration 600

# 输出详细报告
python memory_debug_cli.py leak --output leak_report.json --duration 300
```

#### 4. 碎片分析
```bash
# 分析碎片化并生成图表
python memory_debug_cli.py fragment --duration 120 --plot --output fragment.json

# 持续监控碎片化趋势
python memory_debug_cli.py fragment --duration 300 --interval 5.0
```

#### 5. 分布式监控
```bash
# 启动master节点
python memory_debug_cli.py distributed --role master --port 29500

# 启动worker节点（连接到master）
python memory_debug_cli.py distributed --role worker --master-host 192.168.1.100 --port 29500
```

### 交互式模式
```bash
# 进入交互式调试模式
python memory_debug_cli.py interactive
```

## 📊 使用场景

### 场景1：训练过程中频繁OOM
```bash
# 1. 首先分析当前内存使用模式
python memory_debug_cli.py profile --duration 60 --plot

# 2. 启动OOM检测，设置较低阈值进行预警
python memory_debug_cli.py oom --threshold 75 --monitor

# 3. 如果怀疑内存泄漏，运行泄漏检测
python memory_debug_cli.py leak --enable-tracking --duration 600
```

### 场景2：显存碎片导致的性能下降
```bash
# 1. 分析碎片化程度
python memory_debug_cli.py fragment --duration 180 --plot

# 2. 监控碎片化趋势
python memory_debug_cli.py fragment --duration 600 --interval 10
```

### 场景3：分布式训练负载不均衡
```bash
# 在master节点上启动
python memory_debug_cli.py distributed --role master --duration 600

# 在worker节点上启动
python memory_debug_cli.py distributed --role worker --master-host <master_ip>
```

## 🔧 高级用法

### 编程接口使用

```python
from tools.memory import (
    MemoryProfiler, 
    OOMDetector, 
    FragmentationAnalyzer,
    MemoryLeakDetector,
    DistributedMemoryMonitor
)

# 创建内存分析器
profiler = MemoryProfiler(sampling_interval=1.0)
profiler.start_monitoring()

# 运行训练代码
# ... your training code ...

# 停止监控并生成报告
profiler.stop_monitoring()
report = profiler.generate_report("training_memory_report.json")
```

### 自定义回调函数

```python
def custom_oom_callback(device_id, prediction):
    if prediction.risk_level.value == "critical":
        # 执行紧急操作，如减少batch size
        print(f"Critical OOM risk on GPU {device_id}! Taking action...")
        # torch.cuda.empty_cache()
        # reduce_batch_size()

detector = OOMDetector()
detector.add_warning_callback(custom_oom_callback)
detector.start_monitoring()
```

## 📈 报告格式

### 内存分析报告示例
```json
{
  "timestamp": "2025-06-23T10:30:00",
  "devices": {
    "0": {
      "current_memory": {
        "allocated": 8589934592,
        "free": 2147483648,
        "total": 10737418240,
        "utilization": 80.0
      },
      "peak_memory": 9663676416,
      "statistics": {
        "avg_utilization": 75.5,
        "max_utilization": 95.2,
        "avg_fragmentation": 15.3,
        "num_snapshots": 3600
      }
    }
  }
}
```

### OOM风险报告示例
```json
{
  "devices": {
    "0": {
      "risk_assessment": {
        "risk_level": "high",
        "confidence": 0.8,
        "current_usage_percent": 88.5,
        "growth_rate_mb_per_sec": 12.3,
        "predicted_oom_time_seconds": 45,
        "recommendation": "Reduce batch size immediately"
      }
    }
  }
}
```

## ⚡ 性能优化建议

### 显存优化策略
1. **基于分析结果调整batch size**
2. **使用gradient checkpointing减少峰值内存**
3. **定期清理CUDA缓存**: `torch.cuda.empty_cache()`
4. **使用混合精度训练**: 减少内存使用
5. **优化数据加载**: 避免数据预处理占用过多内存

### 碎片化优化
1. **避免频繁的小块分配释放**
2. **使用内存池**: 预分配大块内存
3. **定期碎片整理**: 在合适的时机清理缓存
4. **优化张量生命周期**: 及时释放不需要的张量

### 分布式优化
1. **负载均衡**: 根据监控结果调整各节点负载
2. **同步优化**: 避免内存使用不均衡导致的同步等待
3. **故障恢复**: 基于监控数据实现智能故障恢复

## 🔍 故障排查

### 常见问题及解决方案

#### 1. "nvidia-smi command not found"
- 确保安装了NVIDIA驱动
- 检查PATH环境变量是否包含nvidia-smi

#### 2. 监控数据不准确
- 检查GPU设备是否正确识别
- 确认监控进程有足够权限
- 验证采样间隔设置是否合理

#### 3. 分布式监控连接失败
- 检查网络connectivity
- 确认防火墙设置
- 验证端口是否被占用

#### 4. 内存泄漏误报
- 调整泄漏检测阈值
- 检查是否有正常的内存增长（如模型权重更新）
- 启用引用跟踪获取详细信息

## 📝 最佳实践

### 1. 监控策略
- **开发阶段**: 使用较短的采样间隔(1-2秒)进行详细分析
- **生产环境**: 使用较长的采样间隔(5-10秒)降低监控开销
- **关键训练**: 启用所有监控工具进行全面分析

### 2. 报告管理
- 定期保存监控报告用于趋势分析
- 建立报告归档机制
- 设置自动化报告生成

### 3. 告警配置
- 根据模型和硬件特点调整告警阈值
- 设置多级告警机制
- 集成到现有的监控系统

## 🤝 贡献指南

欢迎贡献代码和建议！请遵循以下步骤：

1. Fork this repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 许可证

本项目采用与Tiny-Torch相同的许可证。

## 🙋‍♂️ 支持

如果遇到问题或有建议，请：
1. 查看本文档的故障排查部分
2. 在GitHub上提交Issue
3. 联系项目维护者

---

**注意**: 这些工具主要用于调试和分析，在生产环境中使用时请评估性能影响。
