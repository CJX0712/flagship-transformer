/*
 * APEX-TRANSFORMER · 长度外推"补洞"基准
 * ------------------------------------------------------------------
 * 用法：node bench-extrap.js [steps]   （steps 默认 4000）
 *
 * 逻辑：只用短序列 L∈[3,7] 训练一次（原生 RoPE），然后在推理期换上
 *   plain / NTK-aware / YaRN 三种 RoPE 缩放策略，看能否把外推失败救回来。
 *   两种策略都是"推理期免重训"的——这正是它们被称为补洞方案的原因。
 *
 * 诚实原则：救回来就写救回来，救不回来就写救不回来。
 */
const T = require('./apex-transformer.js');

const STEPS = parseInt(process.argv[2], 10) || 4000;
const t0 = Date.now();
const el = () => ((Date.now() - t0) / 1000).toFixed(0) + 's';

console.log(`\n=== 长度外推补洞基准 (训练 L∈[3,7]，steps=${STEPS}) ===`);
console.log('推理期策略：plain = 原生RoPE | ntk = NTK-aware 缩放 | yarn = NTK-by-parts + 注意力温度\n');

const r = T.runExtrapolationFix({ steps: STEPS, testLens: [8, 12, 20, 30] });

console.log('长度  | plain  | ntk    | yarn   | 无注意力基线 | 最佳策略');
console.log('------|--------|--------|--------|--------------|--------');
Object.keys(r.perLen).forEach((L) => {
  const p = r.perLen[L];
  const best = ['plain', 'ntk', 'yarn'].reduce((a, k) => (p[k].acc > p[a].acc ? k : a), 'plain');
  console.log(`L=${String(L).padEnd(4)}| ${p.plain.acc.toFixed(2)}   | ${p.ntk.acc.toFixed(2)}   | ${p.yarn.acc.toFixed(2)}   | ${p.baseline.acc.toFixed(2)}          | ${best}`);
});

const maxL = Math.max.apply(null, Object.keys(r.perLen).map(Number));
console.log(`\n最长未见长度 L=${maxL}：最佳策略 = ${r.bestStrategy}（acc=${r.perLen[maxL][r.bestStrategy].acc.toFixed(2)}，plain=${r.perLen[maxL].plain.acc.toFixed(2)}，基线=${r.perLen[maxL].baseline.acc.toFixed(2)}）`);
console.log(`补洞是否成功: ${r.fixed ? '✅ 是——外推被显著救回' : '⚠️ 否——该策略在此设置下未能救回外推'}`);
console.log(`\n耗时 ${el()}\n`);
process.exit(0);
