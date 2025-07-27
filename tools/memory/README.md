# Memory Debug Tools for Tiny Torch
# Memory Debug Tools

This is a comprehensive memory debugging toolkit designed specifically for Tiny Torch, helping developers troubleshoot and resolve GPU memory issues in distributed training.

## Main Features

### Unified Interface
- **MemoryDebugger**: Unified debugging interface integrating all features
- **Command-line tools**: Support for `python -m tools.memory` invocation
- **Simplified configuration**: Out-of-the-box, minimal configuration

### Core Features
1. **GPU memory analysis** - Real-time monitoring of GPU memory usage, generate detailed reports
2. **OOM detection** - Intelligent warning system, predict OOM occurrence time
3. **Memory leak detection** - Detect GPU memory and Python memory leaks
4. **Fragmentation analysis** - Analyze GPU memory fragmentation (basic version)
5. **Distributed monitoring** - Cross-node GPU memory coordination monitoring (extended feature)

## Quick Start

### Install Dependencies
```bash
pip install numpy matplotlib psutil
# or
pip install -r requirements.txt
```

### Python API Usage

#### 1. Unified Interface (Recommended)
```python
from tools.memory import MemoryDebugger

# Start comprehensive monitoring
debugger = MemoryDebugger()
debugger.start_monitoring()

# Training code...
# your_training_code()

# Stop monitoring and get report
debugger.stop_monitoring()
report = debugger.get_status_report()
print(f"Monitoring summary: {report}")
```

#### 2. Use Individual Tools
```python
from tools.memory import MemoryProfiler, OOMDetector

# Memory analysis
profiler = MemoryProfiler(sampling_interval=0.1)
profiler.start_monitoring()
# ... after running for a while
profiler.stop_monitoring()
profiler.generate_report('memory_report.json')

# OOM detection
detector = OOMDetector(threshold=85.0)
detector.start_monitoring()
# ... automatic warnings
```

### Command Line Usage

#### 1. Memory Analysis
```bash
# Monitor for 60 seconds and generate report
python -m tools.memory profile --duration 60 --output profile.json

# Continuously monitor specific GPUs
python -m tools.memory profile --devices 0 1 --duration 0
```

#### 2. OOM Monitoring
```bash
# OOM monitoring with 85% threshold
python -m tools.memory oom --threshold 85 --monitor

# Monitor specific devices
python -m tools.memory oom --devices 0 --threshold 90
```

#### 3. Memory leak detection
```bash
# Check for memory leaks
python -m tools.memory leak --check --threshold 100
```

#### 4. Comprehensive Monitoring
```bash
# Start all monitoring features
python -m tools.memory monitor --all

# Monitor for specified duration
python -m tools.memory monitor --all --duration 300
```

## Output Examples

### Memory Status Report
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
[WARNING] OOM Warning: GPU 0
   Risk Level: high
   Current Usage: 87.3%
   Predicted OOM Time: 45.2seconds later
   Confidence: 0.82
   Recommendation: Prepare to release cache
   Recommendation: Consider reducing batch size
```

## Advanced Configuration

### Custom OOM Callbacks
```python
def custom_oom_callback(prediction):
    if prediction.risk_level.value == "critical":
        # Execute emergency operations, such as reducing batch size
        print(f"Critical OOM risk on GPU {prediction.device_id}! Taking action...")
        # tiny_torch.cuda.empty_cache()
        # reduce_batch_size()

detector = OOMDetector(warning_callback=custom_oom_callback)
detector.start_monitoring()
```

### GPU Memory Optimization Strategies
1. **Adjust batch size based on analysis results**
2. **Use gradient checkpointing to reduce peak memory**
3. **Regularly clear CUDA cache**: `tiny_torch.cuda.empty_cache()`
4. **Use mixed precision training**: reduce memory usage
5. **Optimize data loading**: avoid data preprocessing taking up too much memory

## Technical Details

### Architecture Design
- **Modular design**: each function is independent and can be used separately
- **Unified interface**: MemoryDebugger provides one-stop solution
- **Lightweight**: minimize impact on training performance
- **Extensible**: Support for自定义回调和策略

### Dependencies
- **numpy**: numerical computation and statistical analysis
- **matplotlib**: generate memory usage charts
- **psutil**: system process monitoring
- **Standard library**: threading, json, subprocess等

### Performance Impact
- Default sampling interval: 0.1秒（adjustable）
- Memory overhead: approximately per device1-5MBhistorical data
- CPUusage: usually < 1%

## Troubleshooting

### Common Issues
1. **nvidia-smi not available**: check NVIDIA driver installation
2. **import error**: ensure required dependencies are installed
3. **permission issues**: some system monitoring features require appropriate permissions

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

debugger = MemoryDebugger()
debugger.start_monitoring()
```

## Version History

### v1.0.0
- Integrate all features into a single file
- Simplify API and command line interface
- Optimize performance and stability
- Improve documentation and examples

## Contributing

Welcome to submit Issues and Pull Requests to improve this toolkit.

## License

This project follows the same license as the main Tiny Torch project.
