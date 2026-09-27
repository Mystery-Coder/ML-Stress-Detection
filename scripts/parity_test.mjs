/**
 * Compare src/lib/inference/mfcc.ts to the librosa reference from parity_test.py.
 *
 *   python scripts/parity_test.py
 *   node --experimental-strip-types scripts/parity_test.mjs
 *
 * Prints Pearson correlation of the mean-over-coefficient MFCC vectors.
 * Gate: r >= 0.99 proceeds; otherwise retrain with this extractor.
 */
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const referencePath = path.join(root, "scripts", "fixtures", "mfcc_reference.json");
const PASS = 0.99;

function tone(sampleRate, seconds, amplitude, frequencyHz) {
  const count = Math.floor(sampleRate * seconds);
  const samples = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    samples[i] = amplitude * Math.sin((2 * Math.PI * frequencyHz * i) / sampleRate);
  }
  return samples;
}

function pearson(a, b) {
  const n = Math.min(a.length, b.length);
  let meanA = 0;
  let meanB = 0;
  for (let i = 0; i < n; i++) {
    meanA += a[i];
    meanB += b[i];
  }
  meanA /= n;
  meanB /= n;
  let num = 0;
  let denA = 0;
  let denB = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i] - meanA;
    const db = b[i] - meanB;
    num += da * db;
    denA += da * da;
    denB += db * db;
  }
  return num / Math.sqrt(denA * denB);
}

const reference = JSON.parse(await readFile(referencePath, "utf8").catch(() => {
  console.error(`No reference at ${referencePath}`);
  console.error("Run: python scripts/parity_test.py");
  process.exit(2);
}));

const mfccUrl = pathToFileURL(path.join(root, "src", "lib", "inference", "mfcc.ts")).href;
let extractMfccMeanOverCoeffs;
try {
  ({ extractMfccMeanOverCoeffs } = await import(mfccUrl));
} catch (error) {
  console.error("Could not import src/lib/inference/mfcc.ts");
  console.error("Run: node --experimental-strip-types scripts/parity_test.mjs");
  console.error(error instanceof Error ? error.message : error);
  process.exit(2);
}

const samples = tone(
  reference.sr,
  reference.seconds,
  reference.amplitude,
  reference.frequency_hz,
);
const js = extractMfccMeanOverCoeffs(samples, reference.sr, {
  nMfcc: reference.n_mfcc,
  nFft: reference.n_fft,
  hopLength: reference.hop_length,
  nMels: reference.n_mels,
  offsetSec: reference.offset,
  durationSec: reference.duration,
});
const py = reference.mean_mfcc;
const r = pearson(js, py);
const sameLength = js.length === py.length;

console.log("MFCC parity (librosa mean axis=0 vs extractMfccMeanOverCoeffs)");
console.log(`  n_fft=${reference.n_fft} hop=${reference.hop_length} n_mels=${reference.n_mels} n_mfcc=${reference.n_mfcc}`);
console.log(`  frames js=${js.length} py=${py.length} match=${sameLength}`);
console.log(`  pearson r = ${r.toFixed(6)}`);
if (!sameLength || !Number.isFinite(r) || r < PASS) {
  console.log(`  decision: FAIL (need r >= ${PASS} and equal frame counts)`);
  console.log("  Mitigation: retrain in Colab using this JS extractor.");
  process.exit(1);
}
console.log(`  decision: PASS (r >= ${PASS})`);
