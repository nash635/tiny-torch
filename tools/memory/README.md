# Memory Debug Tools for Tiny Torch
# 显存调试工具集

这是一套专为 Tiny Torch 设计的显存调试工具，帮助开发者排查和解决分布式训练中的显存问题。

## 🎯 主要功能

### 统一接口
- **MemoryDebugger**: 统一的调试接口，整合所有功能
- **命令行工具**: 支持 `python -m tools.memory` 调用
- **简化配置**: 开箱即用，最小化配置

### 核心功能
1. **显存分析** - 实时监控GPU显存使用情况，生成详细报告
2. **OOM检测** - 智能预警系统，预测OOM发生时间
3. **内存泄漏检测** - 检测显存和Python内存泄漏
4. **碎片分析** - 分析显存碎片化程度（基础版本）
5. **分布式监控** - 跨节点显存使用协调监控（扩展功能）

## 🚀 快速开始

### 安装依赖
```bash
pip install numpy matplotlib psutil
# 或者
pip install -r requirements.txt
```

### Python API 使用

#### 1. 统一接口（推荐）
```python
from tools.memory import MemoryDebugger

# 启动全面监控
debugger = MemoryDebugger()
debugger.start_monitoring()

# 训练代码...
# your_training_code()

# 停止监控并获取报告
debugger.stop_monitoring()
report = debugger.get_status_report()
print(f"监控摘要: {report}")
```

#### 2. 单独使用各工具
```python
from tools.memory import MemoryProfiler, OOMDetector

# 内存分析
profiler = MemoryProfiler(sampling_interval=0.1)
profiler.start_monitoring()
# ... 运行一段时间后
profiler.stop_monitoring()
profiler.generate_report('memory_report.json')

# OOM检测
detector = OOMDetector(threshold=85.0)
detector.start_monitoring()
# ... 自动预警
```

### 命令行使用

#### 1. 内存分析
```bash
# 监控60秒并生成报告
python -m tools.memory profile --duration 60 --output profile.json

# 持续监控特定GPU
python -m tools.memory profile --devices 0 1 --duration 0
```

#### 2. OOM监控
```bash
# 85%阈值OOM监控
python -m tools.memory oom --threshold 85 --monitor

# 监控特定设备
python -m tools.memory oom --devices 0 --threshold 90
```

#### 3. 内存泄漏检测
```bash
# 检查内存泄漏
python -m tools.memory leak --check --threshold 100
```

#### 4. 全面监控
```bash
# 启动所有监控功能
python -m tools.memory monitor --all

# 监控指定时长
python -m tools.memory monitor --all --duration 300
```

## 📊 输出示例

### 内存状态报告
```json
{
  "timestamp": "2025-01-15T10:30:00",
  "devices": {
    "0": {
      "current_usage": {
        "allocated": 8589934592,
        "free": 3758096384,
        "total": 12348030976,
        "utilization_percent": 69.5
      },
      "statistics": {
        "avg_utilization": 65.2,
        "max_utilization": 89.1,
        "samples_count": 600
      }
    }
  },
  "summary": {
    "total_devices": 1,
    "total_memory_gb": 11.5,
    "overall_utilization": 69.5
  }
}
```

### OOM预警示例
```
⚠️  OOM警告: GPU 0
   风险等级: high
   当前使用率: 87.3%
   预测OOM时间: 45.2秒后
   置信度: 0.82
   建议: 准备释放缓存
   建议: 考虑减少batch size
```

## 🔧 高级配置

### 自定义OOM回调
```python
def custom_oom_callback(prediction):
    if prediction.risk_level.value == "critical":
        # 执行紧急操作，如减少batch size
        print(f"Critical OOM risk on GPU {prediction.device_id}! Taking action...")
        # tiny_torch.cuda.empty_cache()
        # reduce_batch_size()

detector = OOMDetector(warning_callback=custom_oom_callback)
detector.start_monitoring()
```

### 显存优化策略
1. **基于分析结果调整batch size**
2. **使用gradient checkpointing减少峰值内存**
3. **定期清理CUDA缓存**: `tiny_torch.cuda.empty_cache()`
4. **使用混合精度训练**: 减少内存使用
5. **优化数据加载**: 避免数据预处理占用过多内存

## 📋 技术说明

### 架构设计
- **模块化设计**: 每个功能独立，可单独使用
- **统一接口**: MemoryDebugger 提供一站式解决方案
- **轻量级**: 最小化对训练性能的影响
- **可扩展**: 支持自定义回调和策略

### 依赖说明
- **numpy**: 数值计算和统计分析
- **matplotlib**: 生成内存使用图表
- **psutil**: 系统进程监控
- **标准库**: threading, json, subprocess等

### 性能影响
- 默认采样间隔: 0.1秒（可调节）
- 内存开销: 每设备约1-5MB历史数据
- CPU占用: 通常 < 1%

## 🔍 故障排除

### 常见问题
1. **nvidia-smi 不可用**: 检查NVIDIA驱动安装
2. **导入错误**: 确保安装了所需依赖
3. **权限问题**: 某些系统监控功能需要适当权限

### 调试模式
```python
import logging
logging.basicConfig(level=logging.DEBUG)

debugger = MemoryDebugger()
debugger.start_monitoring()
```

## 📝 版本历史

### v1.0.0
- 整合所有功能到单一文件
- 简化API和命令行接口
- 优化性能和稳定性
- 完善文档和示例

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个工具集。

## 📄 许可证

本项目遵循与 Tiny Torch 主项目相同的许可证。

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
        # tiny_torch.cuda.empty_cache()
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
3. **定期清理CUDA缓存**: `tiny_torch.cuda.empty_cache()`
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
