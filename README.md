# APEX-TRANSFORMER · 旗舰 Transformer 参考实现 + 基准对照

> 方案2 + 方案1 交付物：把"最先进"变成**有对手、有打分**的实物，并**把尺子往上移一格**。
> 零依赖 · 反向模式自动微分 · RoPE 位置编码 · 归纳任务注意力碾压无注意力基线。

作者：晨星（GitHub: [CJX0712](https://github.com/CJX0712)） · 许可：MIT
地基：[flagship-algo-forge](https://github.com/CJX0712/flagship-algo-forge)（复用其初始化/随机数/注意力数学）

---

## 一、核心结论（真实测量，非宣称）

### ① 短序列归纳：注意力碾压基线
任务：序列 `[c0..c7, c0]` → 预测 `c1`（d=48 / 3 层，2000 步）

| 模型 | loss | acc |
|------|------|-----|
| **注意力 Transformer** | **0.004** | **1.00** |
| 无注意力基线（平均池化 + 线性） | 2.295 | 0.20 |

### ② 长上下文（同分布）：训练即含长序列，注意力依然碾压
任务：训练 L∈[15,20]，测试 L∈{17,20}（d=64 / 4 层，10000 步）

| 测试长度 | 注意力 Transformer | 无注意力基线 | 结论 |
|---------|--------------------|--------------|------|
| L=17 | **acc 0.90** / loss 0.38 | acc 0.14 / loss 2.50 | ✅ 碾压 |
| L=20 | **acc 0.89** / loss 0.67 | acc 0.17 / loss 2.52 | ✅ 碾压 |

### ③ 长度外推：**这是真实短板，不修饰**
任务：训练仅 L∈[3,7]，测试**从未见过**的更长长度

| 测试长度 | 注意力 Transformer | 无注意力基线 | 结论 |
|---------|--------------------|--------------|------|
| L=8 | acc 0.49 | acc 0.15 | ⚠️ 优势在衰减 |
| L=12 | acc 0.26 | acc 0.19 | ⚠️ 优势在衰减 |
| L=20 | acc 0.16 | acc 0.18 | ❌ 崩塌到基线 |
| L=30 | acc 0.14 | acc 0.17 | ❌ 崩塌到基线 |

**结论**：本实现（RoPE + 小模型 + 短序列训练）**能处理长序列，但不会外推到未见长度**。
这不是 bug，是尺子测出来的真实边界。任何声称"我的架构长度泛化很好"的说法，都该先过这一关。

---

## 二、十层架构分解（L1→L10）

| 层 | 名称 | 实现 |
|----|------|------|
| L1 | 数据/分词 | `genExampleInduction` / `genExampleInductionVar` |
| L2 | 张量/嵌入 | `embedding(ids, E)` |
| L3 | 位置编码 | **RoPE** 旋转位置编码（作用于 Q/K，注入相对位置） |
| L4 | 注意力 | 因果多头缩放点积 `multiHeadAttention` |
| L5 | 前馈 MLP | GeLU 扩张-收缩 `W1→GeLU→W2` |
| L6 | 残差/预归一化 | `x = x + f(rmsNorm(x))` |
| L7 | 层堆叠 | N 个预归一化块（长上下文实验用 4 层） |
| L8 | 输出头 | `matmul(xLast, Wlm) + blm` |
| L9 | 损失 | 数值稳定交叉熵 + 反向自动微分 |
| L10 | 评估/基准 | `runBenchmark` / `runGeneralizationBenchmark` |

---

## 三、自检（9/9 全绿）

```
=== APEX-TRANSFORMER 自检 ===
✅ matmul 梯度                  | maxRelErr=5.00e-13
✅ rmsNorm 梯度                 | maxRelErr=4.44e-12
✅ crossEntropy 梯度            | maxRelErr=8.43e-10
✅ RoPE 梯度                    | maxRelErr=8.33e-12
✅ 因果注意力 位置0不看未来              | out[0] 受V[3]扰动=0.00e+0
✅ 注意力 输出有限 & 因果              | finite=true
✅ Transformer 前向             | logits长度=16 finite=true
✅ 基准 注意力碾压基线(归纳)             | TF loss=0.004/acc=1.00 | 基线 loss=2.295/acc=0.20
✅ 可复现性(同种子)                  | ΔTF=0.00e+0
```

每条自检都是一条**可被独立实现交叉验证的数学不变量**：核心算子梯度用中心差分交叉验证（maxRelErr ~1e-10），因果性用"扰动未来不改变过去"验证，RoPE 反向用"逆旋转"验证，可复现性用同种子双跑验证。

---

## 四、站在世界顶级成果之上（出处）

| 组件 | 出处 |
|------|------|
| 缩放点积 / 多头注意力 | Vaswani et al. 2017 (arXiv:1706.03762) |
| RoPE 旋转位置编码 | Su et al. 2021 (arXiv:2104.09864)（Llama/RoPE 系） |
| RMSNorm（预归一化） | Zhang & Sennrich 2019 (arXiv:1910.07467) |
| 层堆叠/前馈（LLaMA/PaLM 系） | Touvron et al. 2023 (arXiv:2302.13971) |
| AdamW（解耦权重衰减） | Loshchilov & Hutter 2019 (arXiv:1904.09237) |
| Xavier / Kaiming 初始化 | Glorot 2010 / He 2015 (arXiv:1502.01852) |
| 反向模式自动微分 | 思想源自 autograd / JAX / tinygrad |
| 梯度裁剪 + LR 衰减 | Pascanu et al. 2013 (arXiv:1211.5063) 等标准训练实践 |

---

## 五、快速开始

```bash
# 快速自检（9/9 全绿，约 1–2 分钟）
node apex-transformer.js

# 长上下文 / 长度外推 专项基准（约 7 分钟，深度实验）
node bench-long.js 10000

# 浏览器可视化自检 + 基准按钮
# 直接用浏览器打开 index.html
```

```js
const T = require('./apex-transformer.js');

// 短序列归纳基准
console.log(T.runBenchmark({ steps: 2000 }));

// 长度泛化基准（可指定训练长度区间与测试长度）
console.log(T.runGeneralizationBenchmark({ trainLmin: 3, trainLmax: 7, testLens: [12, 20] }));

// 仅跑核心算子自检（秒回，不含基准）
T.runSelfTest({ skipBenchmark: true });
```

**环境要求**：仅 Node.js（或现代浏览器）。零依赖、零构建、开箱即跑。

---

## 六、认知反缴械声明

- 本库拒绝"看起来很完整"的幻觉：注意力是否真的有用，由**归纳任务的量化差距**裁决，而非宣称。
- 尺子的价值已经兑现过一次：早期用 KV 检索任务时，**注意力与基线打平（0.50 vs 0.55）**——因为该任务存在"键重复"泄漏捷径。没有基准，这个假优势会被当成真优势一直传下去。
- 这是**玩具规模的参考实现**：验证架构归纳偏置与训练闭环正确，**不宣称**在真实 LLM 基准（GPU 规模）上达到 SOTA。
- 已明确记录的短板：**长度外推失败**（见 §1③）。想补这块，方向是位置编码外推策略（NTK/YaRN 等）与长度感知的训练分布——不是靠调参掩盖。
