import Link from "next/link";
import { Nav } from "@/components/Nav";
import { createClient } from "@/lib/supabase/server";

export const dynamic = "force-dynamic";

export default async function DashboardPage() {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  const { data: tests } = await supabase
    .from("tests")
    .select("id, status, started_at, completed_at, audio_path")
    .order("started_at", { ascending: false });

  return (
    <>
      <Nav email={user?.email} />
      <main className="mx-auto max-w-4xl px-8 py-10">
        <div className="mb-6 flex items-end justify-between">
          <div>
            <h1 className="font-[family-name:var(--font-display)] text-2xl text-[#0f172a]">History</h1>
            <p className="text-xs text-[#94a3b8]">{user?.email}</p>
          </div>
          <Link
            href="/test"
            className="rounded-lg bg-[#1D9E75] px-3 py-[5px] text-[11px] font-medium text-white no-underline hover:bg-[#0F6E56]"
          >
            New recording
          </Link>
        </div>
        <ul className="space-y-2">
          {(tests ?? []).length === 0 ? (
            <li className="rounded-xl border border-[#e2e8f0] bg-white p-4 text-sm">
              No tests yet. Take a recording to create a sealed result.
            </li>
          ) : (
            (tests ?? []).map((test) => (
              <li key={test.id} className="rounded-xl border border-[#e2e8f0] bg-white p-4">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-medium text-[#0f172a]">{test.status}</p>
                    <p className="text-[11px] text-[#94a3b8]">
                      {new Date(test.started_at).toLocaleString()}
                    </p>
                  </div>
                  {test.status === "completed" ? (
                    <Link href={`/result/${test.id}`} className="text-xs text-[#1D9E75]">
                      Unlock
                    </Link>
                  ) : (
                    <span className="text-[11px] text-[#94a3b8]">{test.audio_path ? "audio stored" : "no audio"}</span>
                  )}
                </div>
              </li>
            ))
          )}
        </ul>
      </main>
    </>
  );
}
