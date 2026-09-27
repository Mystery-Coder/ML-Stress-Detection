/**
 * RLS denial check for MoodSense tables and the private audio bucket.
 *
 * Anon and a second user must see zero rows and zero objects.
 * The owner must only see rows whose user_id is their own.
 *
 * Env (shell or .env.local — existing process env wins):
 *   NEXT_PUBLIC_SUPABASE_URL
 *   NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY (or NEXT_PUBLIC_SUPABASE_ANON_KEY)
 *   RLS_OWNER_EMAIL / RLS_OWNER_PASSWORD
 *   RLS_OTHER_EMAIL / RLS_OTHER_PASSWORD
 *
 *   node scripts/rls_test.mjs
 */
import { readFileSync, existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createClient } from "@supabase/supabase-js";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
loadEnvLocal(path.join(root, ".env.local"));

const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
const anonKey =
  process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
const ownerEmail = process.env.RLS_OWNER_EMAIL;
const ownerPassword = process.env.RLS_OWNER_PASSWORD;
const otherEmail = process.env.RLS_OTHER_EMAIL;
const otherPassword = process.env.RLS_OTHER_PASSWORD;

const missing = [
  ["NEXT_PUBLIC_SUPABASE_URL", url],
  ["NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY", anonKey],
  ["RLS_OWNER_EMAIL", ownerEmail],
  ["RLS_OWNER_PASSWORD", ownerPassword],
  ["RLS_OTHER_EMAIL", otherEmail],
  ["RLS_OTHER_PASSWORD", otherPassword],
].filter(([, value]) => !value);

if (missing.length) {
  console.error("Missing env: " + missing.map(([name]) => name).join(", "));
  process.exit(2);
}

const tables = ["profiles", "tests", "predictions", "result_seals"];
let failed = false;

function report(name, ok, detail) {
  console.log(`${ok ? "OK  " : "FAIL"} ${name}: ${detail}`);
  if (!ok) failed = true;
}

const anon = createClient(url, anonKey, { auth: { persistSession: false } });
await expectZero(anon, "anon");

const other = createClient(url, anonKey, { auth: { persistSession: false } });
const otherSignIn = await other.auth.signInWithPassword({
  email: otherEmail,
  password: otherPassword,
});
if (otherSignIn.error || !otherSignIn.data.user) {
  report("other sign-in", false, otherSignIn.error?.message ?? "no user");
} else {
  await expectZero(other, "other");
}

const owner = createClient(url, anonKey, { auth: { persistSession: false } });
const ownerSignIn = await owner.auth.signInWithPassword({
  email: ownerEmail,
  password: ownerPassword,
});
if (ownerSignIn.error || !ownerSignIn.data.user) {
  report("owner sign-in", false, ownerSignIn.error?.message ?? "no user");
} else {
  await expectOwner(owner, ownerSignIn.data.user.id);
}

if (failed) {
  console.log("RLS denial test failed");
  process.exit(1);
}
console.log("RLS denial test passed");

async function expectZero(client, label) {
  for (const table of tables) {
    const { count, error } = await client.from(table).select("*", { count: "exact", head: true });
    const visible = error ? 0 : (count ?? 0);
    const denied = visible === 0;
    report(
      `${label} ${table}`,
      denied,
      error ? `0 rows (${error.code ?? "error"})` : `${visible} rows`,
    );
  }
  const listed = await client.storage.from("audio").list("", { limit: 100 });
  const objects = listed.error ? 0 : (listed.data?.length ?? 0);
  report(
    `${label} storage audio`,
    objects === 0,
    listed.error ? `0 objects (${listed.error.message})` : `${objects} objects`,
  );
}

async function expectOwner(client, userId) {
  const tests = await client.from("tests").select("id,user_id");
  if (tests.error) {
    report("owner tests", false, tests.error.message);
  } else {
    const rows = tests.data ?? [];
    const foreign = rows.filter((row) => row.user_id !== userId);
    report(
      "owner tests",
      foreign.length === 0,
      `${rows.length} own rows, ${foreign.length} foreign`,
    );
  }

  const profiles = await client.from("profiles").select("id");
  if (profiles.error) {
    report("owner profiles", false, profiles.error.message);
  } else {
    const rows = profiles.data ?? [];
    const foreign = rows.filter((row) => row.id !== userId);
    report(
      "owner profiles",
      foreign.length === 0,
      `${rows.length} own rows, ${foreign.length} foreign`,
    );
  }

  const listed = await client.storage.from("audio").list(userId, { limit: 100 });
  if (listed.error) {
    report("owner storage audio", false, listed.error.message);
  } else {
    report("owner storage audio", true, `${listed.data?.length ?? 0} objects under own prefix`);
  }
}

function loadEnvLocal(file) {
  if (!existsSync(file)) return;
  const text = readFileSync(file, "utf8");
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eq = trimmed.indexOf("=");
    if (eq < 0) continue;
    const key = trimmed.slice(0, eq).trim();
    let value = trimmed.slice(eq + 1).trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    if (process.env[key] == null || process.env[key] === "") process.env[key] = value;
  }
}
