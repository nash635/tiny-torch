# 分布式训练技术规范

> **项目目标**: 在分布式多机多卡环境下正确训练ResNet50
> 
> 详细的开发路线图请参考根目录的 [README.md](../README.md#开发路线图)

## 技术架构设计

### 1. 分布式通信架构
```cpp
// csrc/distributed/
├── ProcessGroup.hpp     // 进程组抽象
├── NCCLBackend.cpp     // NCCL通信后端
├── AllReduce.cu        // 梯度同步内核
└── DistributedAPI.cpp  // Python绑定接口
```

### 2. 数据并行实现
```python
# tiny_torch/nn/parallel/
├── distributed.py      // DistributedDataParallel
├── data_parallel.py    // DataParallel (单机多卡)
└── utils.py           // 辅助工具函数
```

### 3. ResNet50目标架构
```python
# examples/resnet/
├── model.py           // ResNet50实现
├── train_distributed.py  // 分布式训练脚本
├── imagenet_dataset.py   // ImageNet数据加载
└── launch_script.sh      // 多机启动脚本
```

## 实施时间计划

### 短期目标 (3个月)
- 完成Phase 2-3的核心实现
- 实现单机多卡的简单模型训练

### 中期目标 (6个月)  
- 完成Phase 4-5的分布式框架
- 实现ResNet18/34的分布式训练

### 长期目标 (12个月)
- 完成完整的ResNet50分布式训练
- 性能达到PyTorch 80%以上效率

## 验证标准

### 功能验证
- [ ] 单机单卡ResNet50训练收敛
- [ ] 单机多卡ResNet50训练收敛  
- [ ] 多机多卡ResNet50训练收敛
- [ ] ImageNet Top-1准确率 > 75%

### 性能验证
- [ ] 通信效率 > 80% (相比PyTorch)
- [ ] 内存使用合理 (无明显泄漏)
- [ ] 扩展性良好 (支持2-8机16-64卡)

### 可用性验证
- [ ] 简单易用的启动命令
- [ ] 完整的错误处理和恢复
- [ ] 清晰的日志和监控

## 技术参考

### 对标目标
参考nanoGPT的分布式训练实现：
- 使用`torchrun`类似的启动方式
- 支持NCCL_IB_DISABLE等环境变量
- 提供梯度累积、混合精度等特性

### 关键技术栈
- **通信后端**: NCCL (NVIDIA Collective Communications Library)
- **分布式框架**: 类似PyTorch DDP的实现
- **数据处理**: DistributedSampler, 多进程数据加载
- **模型并行**: 支持大模型的分片和管道并行

## 计划变更说明

### 主要调整
1. **Phase 5专门化**: 独立的分布式训练框架阶段
2. **Phase 7新增**: ResNet50验证阶段作为最终目标
3. **技术栈增强**: Phase 3增加分布式梯度同步，Phase 4明确ResNet组件

### 实施优先级
1. **核心特性**: 分布式功能作为项目核心，而非附加功能
2. **验证驱动**: 以ResNet50实际训练为最终验证标准
3. **渐进实现**: 先单机多卡，再多机多卡
4. **性能导向**: 重视通信效率和扩展性
