/*
 * APEX-TRANSFORMER · ⑤ 长上下文预训练基准（真正补外推）
 * ------------------------------------------------------------------
 * 用法：node bench-longctx.js [steps] [testLens] [curric] [cfg]
 *   steps    默认 4000
 *   testLens 默认 "30,40"（逗号分隔，单位：token）
 *   curric   默认 "3-7,8-13,14-20"（长度课程，逗号分隔，每个 "min-max"）
 *   cfg      默认 "d48L3F192"（d=维度 heads=层 ffn=前馈）固定 heads=layers
 *
 * 对照两种 regime 在【相同未见长度】上的表现：
 *   A. short(+YaRN/原生) —— 旧补洞方案（训练 L∈[3,7]，推理期可选 YaRN 缩放）
 *   B. longCurric        —— ⑤ 长上下文预训练（长度课程逐段拉长）
 *        · longPlain : 推理期原生 RoPE（不再靠缩放补丁）
 *        · longYaRN  : 推理期对超出训练上限的长度再叠 YaRN（双保险）
 *
 * 诚实原则：同时报告三条标准——
 *   (a) 原生外推是否被训练修复（长训原生 > 短训原生）
 *   (b) 组合拳是否真正超越旧补丁（长训+YaRN > 短训+YaRN）
 *   (c) 旧极严判据（长训原生 > 短训补丁）仅作参照
 */
const T = require('./apex-transformer.js');
const fs = require('fs');

const STEPS = parseInt(process.argv[2], 10) || 4000;
const TESTLENS = (process.argv[3] || '30,40').split(',').map((x) => parseInt(x, 10));

function parseCurric(s) {
  return (s || '3-7,8-13,14-20').split(',').map((seg) => seg.split('-').map((x) => parseInt(x, 10)));
}
function parseCfg(s) {
  // 形如 "d48L3F192" → {d:48, heads:3, layers:3, ffn:192}
  const m = (s || 'd48L3F192').match(/d(\d+)L(\d+)F(\d+)/i);
  if (!m) return { d: 48, heads: 3, layers: 3, ffn: 192 };
  const d = +m[1], L = +m[2], f = +m[3];
  return { d, heads: L, layers: L, ffn: f };
}
const CURRIC = parseCurric(process.argv[4]);
const CFG = parseCfg(process.argv[5]);
const LR = parseFloat(process.argv[6]) || 3e-3;
const curricStr = CURRIC.map((c) => `[${c[0]},${c[1]}]`).join('→');
const t0 = Date.now();
const el = () => ((Date.now() - t0) / 1000).toFixed(0) + 's';

console.log(`\n=== ⑤ 长上下文预训练基准 (steps=${STEPS}, 测试未见长度=${TESTLENS.join('/')}, cfg d=${CFG.d}/L${CFG.layers}, lr=${LR}) ===`);
console.log(`长度课程: ${curricStr}（最长训练长度=${CURRIC[CURRIC.length - 1][1]}）\n`);

const r = T.runLongContextBenchmark({ steps: STEPS, testLens: TESTLENS, cfg: CFG, curric: CURRIC, lr: LR });

const pad = (n) => (n >= 0 ? ' ' : '') + n.toFixed(2);
console.log('长度  | 短训原生 | 短训+YaRN | 长训原生 | 长训+YaRN | 结论');
console.log('------|----------|-----------|----------|-----------|------');
Object.keys(r.perLen).forEach((L) => {
  const p = r.perLen[L];
  const verdict = p.longPlain.acc > p.shortPlain.acc + 0.10 ? '✅ 长训原生已超短训原生'
    : (p.longYaRN.acc > p.shortYaRN.acc + 0.10 ? '✅ 长训+YaRN超补丁' : '⚠️ 仍需努力');
  console.log(`L=${String(L).padEnd(4)}| ${pad(p.shortPlain.acc)}    | ${pad(p.shortYaRN.acc)}     | ${pad(p.longPlain.acc)}    | ${pad(p.longYaRN.acc)}     | ${verdict}`);
});

const maxL = Math.max.apply(null, TESTLENS);
console.log(`\n最长未见长度 L=${maxL}:`);
console.log(`  短训原生=${r.shortPlainAcc.toFixed(2)}  短训+YaRN=${r.shortAcc.toFixed(2)}`);
console.log(`  长训原生=${r.longPlainAcc.toFixed(2)}  长训+YaRN=${r.longYaRNacc.toFixed(2)}`);
console.log(`\n(a) 原生外推被训练修复（长训原生 > 短训原生 +0.10）：${r.nativeImproved ? '✅ 是' : '⚠️ 否'}`);
console.log(`(b) 组合拳超越旧补丁（长训+YaRN > 短训+YaRN +0.10）：${r.longYaRNbeatsShort ? '✅ 是——⑤ 真正价值兑现' : '⚠️ 否'}`);
console.log(`(c) 旧极严判据（长训原生 > 短训补丁 +0.10）：${r.improved ? '✅ 是' : '⚠️ 否（仅作参照）'}`);
console.log(`\n耗时 ${el()}\n`);

fs.writeFileSync('bench-longctx.result.json', JSON.stringify(r, null, 2));
console.log('结果已写入 bench-longctx.result.json');
process.exit(0);
