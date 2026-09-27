import { createClient } from "jsr:@supabase/supabase-js@2";

const RETENTION_MS = 30 * 24 * 60 * 60 * 1000;
const PAGE = 200;

type TestRow = {
  id: string;
  user_id: string;
  audio_path: string | null;
};

Deno.serve(async () => {
  const url = Deno.env.get("SUPABASE_URL");
  const serviceKey =
    Deno.env.get("SUPABASE_SECRET_KEY") ?? Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
  if (!url || !serviceKey) {
    return Response.json({ error: "missing SUPABASE_URL or SUPABASE_SECRET_KEY" }, { status: 500 });
  }

  const supabase = createClient(url, serviceKey, {
    auth: { persistSession: false, autoRefreshToken: false },
  });

  const cutoff = new Date(Date.now() - RETENTION_MS).toISOString();
  let purged = 0;
  const failures: string[] = [];

  for (let from = 0; ; from += PAGE) {
    const { data, error } = await supabase
      .from("tests")
      .select("id, user_id, audio_path")
      .lt("started_at", cutoff)
      .neq("status", "purged")
      .order("started_at", { ascending: true })
      .range(from, from + PAGE - 1);

    if (error) {
      return Response.json({ error: error.message, purged, failures }, { status: 500 });
    }

    const tests = (data ?? []) as TestRow[];
    if (tests.length === 0) break;

    for (const test of tests) {
      try {
        await removeAudio(supabase, test);
        const { error: updateError } = await supabase
          .from("tests")
          .update({ status: "purged" })
          .eq("id", test.id);
        if (updateError) throw new Error(updateError.message);
        purged += 1;
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        failures.push(`${test.id}: ${message}`);
      }
    }

    if (tests.length < PAGE) break;
  }

  return Response.json({ purged, failures });
});

async function removeAudio(
  supabase: ReturnType<typeof createClient>,
  test: TestRow,
) {
  const folder = `${test.user_id}/${test.id}`;
  const paths = new Set<string>();

  if (test.audio_path) paths.add(test.audio_path);

  const { data, error } = await supabase.storage.from("audio").list(folder, { limit: 1000 });
  if (error) throw new Error(error.message);

  for (const entry of data ?? []) {
    if (entry.id) paths.add(`${folder}/${entry.name}`);
  }

  if (paths.size === 0) return;

  const { error: removeError } = await supabase.storage.from("audio").remove([...paths]);
  if (removeError) throw new Error(removeError.message);
}
