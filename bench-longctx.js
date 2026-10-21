/*
 * APEX-TRANSFORMER · ⑤ 长上下文预训练基准（真正补外推）
 * ------------------------------------------------------------------
 * 用法：node bench-longctx.js [steps] [testLens逗号分隔]
 *   steps 默认 4000；testLens 默认 "30,40"
 *
 * 对照两种 regime 在【相同未见长度】上的表现：
 *   A. short+YaRN  —— 旧补洞方案（训练 L∈[3,7]，推理期 YaRN 缩放）
 *   B. longCurric  —— ⑤ 长上下文预训练（课程 [3,7]→[8,15]→[16,24]）
 *        · longPlain : 推理期原生 RoPE（不再靠缩放补丁）
 *        · longYaRN  : 推理期对超出训练上限的长度再叠 YaRN（双保险）
 *
 * 诚实原则：长训原生 RoPE 须明显优于短训+YaRN，才算"把外推从补丁变成能力"。
 */
const T = require('./apex-transformer.js');
const fs = require('fs');

const STEPS = parseInt(process.argv[2], 10) || 4000;
const TESTLENS = (process.argv[3] || '30,40').split(',').map((x) => parseInt(x, 10));
// ⑤ 用较小模型以保证长课程能充分收敛（与短训补丁同配置对照，公平）；
// 如需与 d=64 外推基准直接对齐，可改此处为大配置。
const CFG = { d: 48, heads: 3, layers: 3, ffn: 192 };
const CURRIC = [[3, 7], [8, 13], [14, 20]];
const t0 = Date.now();
const el = () => ((Date.now() - t0) / 1000).toFixed(0) + 's';

console.log(`\n=== ⑤ 长上下文预训练基准 (steps=${STEPS}, 测试未见长度=${TESTLENS.join('/')}, cfg d=${CFG.d}/L${CFG.layers}) ===`);
console.log('Regime A: 短训 L∈[3,7] + 推理期 YaRN（旧补洞）');
console.log('Regime B: 长度课程 [3,7]→[8,13]→[14,20]（⑤ 长上下文预训练）\n');

const r = T.runLongContextBenchmark({ steps: STEPS, testLens: TESTLENS, cfg: CFG, curric: CURRIC });

const pad = (n) => (n >= 0 ? ' ' : '') + n.toFixed(2);
console.log('长度  | 短训+YaRN(A) | 长训原生(B·plain) | 长训+YaRN(B·yarn) | 结论');
console.log('------|--------------|-------------------|-------------------|------');
Object.keys(r.perLen).forEach((L) => {
  const p = r.perLen[L];
  const verdict = p.longPlain.acc > p.shortYaRN.acc + 0.10 ? '✅ 长训原生已超补丁'
    : (p.longYaRN.acc > p.shortYaRN.acc + 0.10 ? '✅ 长训+YaRN超补丁' : '⚠️ 仍需努力');
  console.log(`L=${String(L).padEnd(4)}| ${pad(p.shortYaRN.acc)}        | ${pad(p.longPlain.acc)}             | ${pad(p.longYaRN.acc)}             | ${verdict}`);
});

const maxL = Math.max.apply(null, TESTLENS);
console.log(`\n最长未见长度 L=${maxL}：短训+YaRN=${r.shortAcc.toFixed(2)} → 长训原生=${r.longPlainAcc.toFixed(2)}（YaRN=${r.longYaRNacc.toFixed(2)}）`);
console.log(`⑤ 是否真正补上外推（长训原生 > 短训补丁 +0.10）：${r.improved ? '✅ 是——外推从"推理期补丁"变成"训出来的能力"' : '⚠️ 否——本设置下长训尚未明显超越补丁'}`);
console.log(`\n耗时 ${el()}\n`);

fs.writeFileSync('bench-longctx.result.json', JSON.stringify(r, null, 2));
console.log('结果已写入 bench-longctx.result.json');
process.exit(0);
