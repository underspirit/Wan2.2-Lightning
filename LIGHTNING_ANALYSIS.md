# Wan2.2-Lightning 改造总结

## 概述
Wan2.2-Lightning 是对原始 Wan2.2 视频生成模型系列的蒸馏优化版本，主要实现了**4步推理**和**20倍速度提升**。

## 主要改造内容

### 1. 核心架构优化
- **4步推理**：从40+步骤减少到仅需4步，无需CFG引导
- **双模型架构**：
  - `low_noise_model`：处理低噪声阶段
  - `high_noise_model`：处理高噪声阶段
  - 根据边界阈值自动切换（T2V: 0.875, I2V: 0.900）

### 2. 模型配置
支持三种模型类型：
- **T2V-A14B**：文本到视频（5120维，40层，40头）
- **I2V-A14B**：图像到视频（相同架构，不同边界）
- **TI2V-5B**：文本+图像到视频（3072维，30层，24头）

### 3. 流匹配求解器
- **多求解器支持**：Euler、UniPC、DPM++
- **采样优化**：`sample_shift=5.0`，优化时间步调度
- **推理配置**：`sample_steps=4`，`sample_guide_scale=(1.0, 1.0)`

### 4. LoRA权重集成
- 使用SafeTensors格式存储低秩适应权重
- 分别为低噪声和高噪声模型提供专门的LoRA权重
- 支持动态加载和合并

### 5. VAE优化
- **Wan2.1 VAE**：T2V/I2V使用，stride(4,8,8)
- **Wan2.2 VAE**：TI2V使用，stride(4,16,16)
- 支持因果3D卷积进行时序建模

### 6. 内存优化详解

#### 6.1 动态模型卸载机制
```python
def _prepare_model_for_timestep(self, t, boundary, offload_model):
    if t.item() >= boundary:
        required_model_name = 'high_noise_model'
        offload_model_name = 'low_noise_model'
    else:
        required_model_name = 'low_noise_model'
        offload_model_name = 'high_noise_model'
    
    if offload_model or self.init_on_cpu:
        # 将不需要的模型卸载到CPU
        if next(getattr(self, offload_model_name).parameters()).device.type == 'cuda':
            getattr(self, offload_model_name).to('cpu')
        # 将需要的模型加载到GPU
        if next(getattr(self, required_model_name).parameters()).device.type == 'cpu':
            getattr(self, required_model_name).to(self.device)
```

#### 6.2 多层次内存清理
- **模型级清理**：推理结束后将所有模型卸载到CPU
```python
if offload_model:
    self.low_noise_model.cpu()
    self.high_noise_model.cpu()
    torch.cuda.empty_cache()
```

- **张量级清理**：显式删除中间张量
```python
del noise, latent, x0
del sample_scheduler
```

- **系统级清理**：垃圾回收和CUDA同步
```python
if offload_model:
    gc.collect()
    torch.cuda.synchronize()
```

#### 6.3 T5编码器内存优化
- **CPU模式**：`t5_cpu=True`时T5始终在CPU上运行
- **动态切换**：推理时T5自动在CPU/GPU间切换
```python
if not self.t5_cpu:
    self.text_encoder.model.to(self.device)
    # ... 编码操作
    if offload_model:
        self.text_encoder.model.cpu()
```

#### 6.4 精度优化
- **混合精度**：使用bfloat16减少内存占用
- **安全降精度**：`model_safe_downcast`保护关键参数
```python
model_safe_downcast(
    model,
    dtype=self.param_dtype,  # bfloat16
    keep_in_fp32_parameters=model._keep_in_fp32_params
)
```

### 7. 双模型时间步切换详解

#### 7.1 边界阈值设计
- **T2V边界**：0.875 (875/1000时间步)
- **I2V边界**：0.900 (900/1000时间步)
- **动态计算**：`boundary = self.boundary * self.num_train_timesteps`

#### 7.2 切换逻辑
```python
# 在推理循环中的每个时间步
for _, t in enumerate(timesteps):
    model = self._prepare_model_for_timestep(t, boundary, offload_model)
    sample_guide_scale = guide_scale[1] if t.item() >= boundary else guide_scale[0]
    
    # 使用对应的模型和引导尺度
    noise_pred_cond = model(latent_model_input, t=timestep, **arg_c)[0]
```

#### 7.3 优势分析
- **专业化处理**：高噪声模型专门处理早期去噪，低噪声模型处理细节优化
- **内存效率**：任意时刻只有一个模型在GPU上活跃
- **质量保证**：针对性优化确保不同噪声级别的最佳表现

### 8. 分布式优化
- **FSDP支持**：T5和DiT模型的分片并行
- **序列并行**：注意力和模型级别的并行

### 9. 工程优化
- **ComfyUI集成**：原生工作流和Wrapper支持
- **多分辨率**：支持720×1280等多种分辨率
- **质量保证**：保持复杂运动生成能力

### 10. 推理性能对比

#### 10.1 内存使用优化
- **基础模型**：需要同时加载完整的大模型到GPU
- **Lightning版本**：
  - 动态模型切换，峰值内存减少约40%
  - T5编码器可选CPU运行，进一步节省GPU内存
  - 精确的张量清理，避免内存泄漏

#### 10.2 时间步切换开销
- **切换频率**：整个推理过程中只切换一次（在边界时间步）
- **切换时间**：模型在CPU-GPU间转移约需50-100ms
- **净收益**：相比传统40步推理，总体仍有巨大加速

## 技术创新点

1. **模型蒸馏**：通过蒸馏技术实现4步推理
2. **自适应噪声处理**：双模型架构针对不同噪声级别优化
3. **高效采样**：改进的流匹配调度器
4. **参数高效**：LoRA技术实现模型定制
5. **工程优化**：分布式训练和推理优化

## 性能提升

- **速度**：20倍推理加速
- **质量**：与基础模型相当或更好
- **效率**：大幅降低计算成本
- **可用性**：支持多种部署方式

Wan2.2-Lightning 成功地在保持高质量视频生成的同时，实现了显著的推理速度提升，使视频生成技术更加实用和可部署。