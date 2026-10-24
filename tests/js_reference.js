// Runs the ORIGINAL trainer coil model (engine.js) and prints reference numbers as JSON.
require(process.argv[2]);               // path to Cracking-Furnace-Trainer/src/engine.js
const F = globalThis.Furnace;
const MWe = 30.07, MWw = 18.02;
const nE0 = 32e6 / 3600 / 4 / MWe, nW0 = 32 * 0.32e6 / 3600 / 4 / MWw;
const cases = [
  [1.0, 1.0, 1.0], [0.8, 1.0, 1.0], [1.0, 0.7, 1.0], [1.0, 1.0, 1.05], [1.1, 1.2, 0.95]
].map(([fe, fw, fq]) => {
  const r = F.solveCoil(nE0 * fe, nW0 * fw, F.CAL0.Q * fq, F.CAL0.V);
  return { fe, fw, fq, cot: r.cot, X: r.X, y: r.y };
});
console.log(JSON.stringify({ CAL0: F.CAL0, cases }));
