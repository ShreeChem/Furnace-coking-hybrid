// Proves the page's live euro calculator gives the same numbers as value.py.
// Runs the page script in a tiny fake DOM and compares with web/data.json's Python reference values.
// Run:  node tests/test_web.js
const fs = require('fs'), path = require('path');
const html = fs.readFileSync(path.join(__dirname, '..', 'web', 'index.html'), 'utf8');
const script = html.split('<script>')[1].split('</script>')[0];
const data = JSON.parse(html.match(/const DATA = (\{.*?\});\n/s)[1]);
// minimal stubs: we only need the pure functions, so cut the script before the drawing/wiring part
const pure = script.split('/* ---------- wiring ---------- */')[0];
const vals = {'s-margin':15, 's-days':2, 's-cost':50, 's-spare':false};
const fake = id => ({ get value(){ return vals[id]; }, get checked(){ return vals[id]; } });
const fn = new Function('document', 'getComputedStyle', pure.replace('const $ = id => document.getElementById(id);', 'const $ = id => document.getElementById(id);') + '; return {caseA, econ};');
const api = fn({getElementById:fake, documentElement:{}}, () => ({getPropertyValue:() => ''}));
let fails = 0;
for (const [key, ref] of Object.entries(data.case_a_reference)) {
  const [model, method, spare] = key.split('|'); vals['s-spare'] = spare === 'True';
  const got = api.caseA(model, api.econ(), method).lost;
  const ok = Math.abs(got - ref) < 1.0;
  if (!ok) fails++;
  console.log(`${ok ? 'OK  ' : 'FAIL'} ${key.padEnd(32)} page ${got.toFixed(0).padStart(7)}  python ${ref.toFixed(0).padStart(7)}`);
}
console.log(fails ? `\n${fails} FAILED` : '\nALL PASSED');
process.exit(fails ? 1 : 0);
