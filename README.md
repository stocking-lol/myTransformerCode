# myTransformerCode
自用记录：学习自己动手写一个Transformer
用Transformer替代CNN在图形分类中的作用，参考模型Vision-Transformer

## 核心思想：
将图像分割为固定大小的图像块（Patch），每个块展平为序列输入Transformer编码器。
完全摒弃传统CNN，依赖自注意力机制捕捉全局关系。

## 结构特点：
注意：因为执行的是图片分类任务，所以并不需要改掉后面的输入，所以不需要掩码Mask，为了起到记录的作用仍然保持了Mask.py，并且保留了有Mask的MutiHeadAttetion

输入处理：图像 → 分割为16x16的块 → 线性嵌入 → 添加位置编码。

分类头：使用一个额外的[CLS] token进行图像分类。

## 优点：
在大规模数据集（如JFT-300M）上训练时，性能超过ResNet等CNN模型。
支持并行计算，适合处理高分辨率图像。

## 局限：
需要大量数据预训练，小数据集表现较差。
计算复杂度高（序列长度为N²，N为块数）。
