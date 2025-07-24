# Memory Debug Tools for Tiny Torch
# 显存调试工具集

这是一套专为 Tiny Torch 设计的显存调试工具，帮助开发者排查和解决分布式训练中的显存问题。

## 主要功能

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

## 快速开始

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

## 输出示例

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
[WARNING] OOM警告: GPU 0
   风险等级: high
   当前使用率: 87.3%
   预测OOM时间: 45.2秒后
   置信度: 0.82
   建议: 准备释放缓存
   建议: 考虑减少batch size
```

## 高级配置

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

## 技术说明

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

## 故障排除

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

## 版本历史

### v1.0.0
- 整合所有功能到单一文件
- 简化API和命令行接口
- 优化性能和稳定性
- 完善文档和示例

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个工具集。

## 许可证

本项目遵循与 Tiny Torch 主项目相同的许可证。
