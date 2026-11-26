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

  const allTests = tests ?? [];
  const completed = allTests.filter((test) => test.status === "completed").length;
  const pending = allTests.length - completed;

  const stats = [
    { label: "Total tests", value: allTests.length },
    { label: "Results ready", value: completed },
    { label: "In progress", value: pending },
  ];

  return (
    <>
      <Nav email={user?.email} />
      <main className="mx-auto w-full max-w-6xl px-6 py-10 sm:px-8 lg:px-10">
        <div className="mb-8 flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="font-[family-name:var(--font-display)] text-3xl text-[#0f172a]">History</h1>
            <p className="mt-1 text-sm text-[#94a3b8]">{user?.email}</p>
          </div>
          <Link
            href="/test"
            className="rounded-lg bg-[#1D9E75] px-4 py-2 text-xs font-medium text-white no-underline hover:bg-[#0F6E56]"
          >
            New recording
          </Link>
        </div>

        <div className="mb-8 grid grid-cols-1 gap-4 sm:grid-cols-3">
          {stats.map((stat) => (
            <div
              key={stat.label}
              className="rounded-2xl border border-[#e2e8f0] bg-white p-6"
            >
              <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-[#94a3b8]">
                {stat.label}
              </p>
              <p className="mt-2 text-3xl font-semibold text-[#0f172a]">{stat.value}</p>
            </div>
          ))}
        </div>

        <ul className="space-y-3">
          {allTests.length === 0 ? (
            <li className="rounded-2xl border border-[#e2e8f0] bg-white p-6 text-sm">
              No tests yet. Take a recording to create a sealed result.
            </li>
          ) : (
            allTests.map((test) => (
              <li key={test.id} className="rounded-2xl border border-[#e2e8f0] bg-white p-5">
                <div className="flex items-center justify-between gap-4">
                  <div className="min-w-0">
                    <p className="text-base font-medium capitalize text-[#0f172a]">{test.status}</p>
                    <p className="mt-1 text-sm text-[#94a3b8]">
                      {new Date(test.started_at).toLocaleString()}
                    </p>
                  </div>
                  {test.status === "completed" ? (
                    <Link
                      href={`/result/${test.id}`}
                      className="shrink-0 rounded-lg border border-[#5DCAA5] bg-white px-3 py-1.5 text-xs font-medium text-[#1D9E75] no-underline hover:bg-[#E1F5EE]"
                    >
                      Unlock
                    </Link>
                  ) : (
                    <span className="shrink-0 text-xs text-[#94a3b8]">
                      {test.audio_path ? "audio stored" : "no audio"}
                    </span>
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
