/*
 * APEX-TRANSFORMER · 长上下文 / 长度外推 专项基准
 * ------------------------------------------------------------------
 * 用法：node bench-long.js [steps]
 *   steps 默认 10000（长上下文 regime 需要足够步数才能收敛）
 *
 * 两个 regime：
 *   A. 长上下文（同分布）：训练 L∈[15,20]，测试 L∈{17,20}
 *      → 验证"架构能不能处理长序列"（训练充分时）
 *   B. 长度外推：训练 L∈[3,7]，测试 L∈{8,12,20,30}
 *      → 验证"能否外推到未见长度"（已知短板，见 README）
 *
 * 诚实原则：两个 regime 都报告真实数字，优势衰减就写衰减，不修饰。
 */
const T = require('./apex-transformer.js');

const STEPS = parseInt(process.argv[2], 10) || 10000;
const t0 = Date.now();
const el = () => ((Date.now() - t0) / 1000).toFixed(0) + 's';

console.log(`\n=== APEX-TRANSFORMER 长上下文专项基准 (steps=${STEPS}) ===\n`);

console.log('A. 长上下文（同分布）训练 L∈[15,20]，测试 L∈{17,20} ...');
const A = T.runGeneralizationBenchmark({ trainLmin: 15, trainLmax: 20, testLens: [17, 20], steps: STEPS });
console.log('   结果：');
Object.keys(A.perLen).forEach((L) => {
  const p = A.perLen[L];
  console.log(`     L=${L}  TF acc=${p.transformer.acc.toFixed(2)} loss=${p.transformer.loss.toFixed(3)}  |  基线 acc=${p.baseline.acc.toFixed(2)} loss=${p.baseline.loss.toFixed(3)}`);
});
console.log(`   注意力碾压基线: ${A.transformerWins ? '✅ 是' : '❌ 否'}\n   [${el()}]\n`);

console.log('B. 长度外推：训练 L∈[3,7]，测试未见更长长度 ...');
const B = T.runGeneralizationBenchmark({ trainLmin: 3, trainLmax: 7, testLens: [8, 12, 20, 30], steps: 4000 });
console.log('   结果：');
Object.keys(B.perLen).forEach((L) => {
  const p = B.perLen[L];
  console.log(`     L=${L}  TF acc=${p.transformer.acc.toFixed(2)} loss=${p.transformer.loss.toFixed(3)}  |  基线 acc=${p.baseline.acc.toFixed(2)} loss=${p.baseline.loss.toFixed(3)}`);
});
const stillWins = B.beatsBaselineEverywhere;
console.log(`   注意力处处仍优于基线: ${stillWins ? '✅ 是' : '⚠️ 否（极端长度已崩塌到基线水平——长度外推是真实短板）'}`);
console.log(`\n总耗时 ${el()}\n`);
process.exit(0);
