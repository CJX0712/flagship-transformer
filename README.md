# APEX-TRANSFORMER · 旗舰 Transformer 参考实现 + 基准对照

> 方案2交付物：把"最先进"变成**有对手、有打分**的实物。
> 零依赖 · 反向模式自动微分 · 归纳任务上注意力碾压无注意力基线。

作者：晨星（GitHub: [CJX0712](https://github.com/CJX0712)） · 许可：MIT
地基：[flagship-algo-forge](https://github.com/CJX0712/flagship-algo-forge)（复用其初始化/随机数/注意力数学）

---

## 一、核心结论（基准结果）

在**归纳头任务**上（序列 `[c0..c7, c0]` → 预测 `c1`），训练 2000 步：

| 模型 | loss | acc |
|------|------|-----|
| **注意力 Transformer**（本实现） | **0.000** | **1.00** |
| 无注意力基线（平均池化 + 线性） | 2.295 | 0.20 |

注意力架构学会"把末尾的 c0 路由回首位置读取后继"；无位置信息的平均池化必然失败（acc≈随机 1/16）。

> 这是给"最先进"的一把尺子：**不是自说自话，而是让架构的归纳偏置在任务上被量化验证。** 若注意力赢不了，说明只是漂亮概念——实测它赢了。

---

## 二、十层架构分解（L1→L10）

| 层 | 名称 | 实现 |
|----|------|------|
| L1 | 数据/分词 | `genExampleInduction` |
| L2 | 张量/嵌入 | `embedding(ids, E)` |
| L3 | 位置编码 | 学习型绝对位置嵌入 `+ P` |
| L4 | 注意力 | 因果多头缩放点积 `multiHeadAttention` |
| L5 | 前馈 MLP | GeLU 扩张-收缩 `W1→GeLU→W2` |
| L6 | 残差/预归一化 | `x = x + f(rmsNorm(x))` |
| L7 | 层堆叠 | N 个预归一化块（默认 3） |
| L8 | 输出头 | `matmul(xLast, Wlm) + blm` |
| L9 | 损失 | 数值稳定交叉熵 + 反向自动微分 |
| L10 | 评估/基准 | `runBenchmark` |

---

## 三、自检（8/8 全绿）

```
=== APEX-TRANSFORMER 自检 ===
✅ matmul 梯度                  | maxRelErr=5.00e-13
✅ rmsNorm 梯度                 | maxRelErr=4.44e-12
✅ crossEntropy 梯度            | maxRelErr=8.43e-10
✅ 因果注意力 位置0不看未来              | out[0] 受V[3]扰动=0.00e+0
✅ 注意力 输出有限 & 因果              | finite=true
✅ Transformer 前向             | logits长度=16 finite=true
✅ 基准 注意力碾压基线(归纳)             | TF loss=0.000/acc=1.00 | 基线 loss=2.295/acc=0.20
✅ 可复现性(同种子)                  | ΔTF=0.00e+0
```

每条自检都是一条**可被独立实现交叉验证的数学不变量**：核心算子梯度用中心差分交叉验证（maxRelErr ~1e-10），因果性用"扰动未来不改变过去"验证，可复现性用同种子双跑验证。

---

## 四、站在世界顶级成果之上（出处）

| 组件 | 出处 |
|------|------|
| 缩放点积 / 多头注意力 | Vaswani et al. 2017 (arXiv:1706.03762) |
| RMSNorm（预归一化） | Zhang & Sennrich 2019 (arXiv:1910.07467) |
| 层堆叠/前馈（LLaMA/PaLM 系） | Touvron et al. 2023 (arXiv:2302.13971) |
| AdamW（解耦权重衰减） | Loshchilov & Hutter 2019 (arXiv:1904.09237) |
| Xavier / Kaiming 初始化 | Glorot 2010 / He 2015 (arXiv:1502.01852) |
| 反向模式自动微分 | 思想源自 autograd / JAX / tinygrad |
| 梯度裁剪 + LR 衰减 | Pascanu et al. 2013 (arXiv:1211.5063) 等标准训练实践 |

---

## 五、快速开始

```bash
# 无头自检 + 基准（8/8 全绿）
node apex-transformer.js

# 浏览器可视化自检 + 基准按钮
# 直接用浏览器打开 index.html
```

```js
const T = require('./apex-transformer.js');

// 训练一个 Transformer 并跑基准
const bench = T.runBenchmark({ steps: 2000 });
console.log(bench.transformer, bench.baseline);

// 仅跑核心算子自检（秒回，不含基准）
T.runSelfTest({ skipBenchmark: true });
```

**环境要求**：仅 Node.js（或现代浏览器）。零依赖、零构建、开箱即跑。

---

## 六、认知反缴械声明

- 本库拒绝"看起来很完整"的幻觉：注意力是否真的有用，由**归纳任务的量化差距**裁决，而非宣称。
- 这是**玩具规模的参考实现**：用来验证架构归纳偏置与训练闭环正确，**不宣称**在真实 LLM 基准（GPU 规模）上达到 SOTA。
- 下一步：以更难的检索/泛化任务 + 更大模型把尺子继续上移，让"最先进"始终有对手、有打分。
