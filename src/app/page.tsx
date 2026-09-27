import Link from "next/link";
import { Nav } from "@/components/Nav";
import { createClient } from "@/lib/supabase/server";

export const dynamic = "force-dynamic";

export default async function HomePage() {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  return (
    <>
      <Nav email={user?.email} />
      <main className="mx-auto max-w-4xl px-8 py-20">
        <div className="rounded-xl border border-[#9FE1CB] bg-[#E1F5EE] p-6">
          <p className="text-[10px] font-semibold uppercase tracking-[0.08em] text-[#0F6E56]">
            MFCC-powered · server-side inference
          </p>
          <h1 className="mt-1 font-[family-name:var(--font-display)] text-[28px] font-semibold leading-tight text-[#0f172a]">
            Detect emotion
            <br />
            through your voice
          </h1>
          <p className="mt-2 max-w-xl text-sm leading-relaxed text-[#475569]">
            Record answers to 42 prompts. Audio stays in a private bucket (access control).
            Predictions are sealed in your browser with a key only you hold.
          </p>
          <div className="mt-4 flex flex-wrap gap-2">
            <Link
              href={user ? "/test" : "/register"}
              className="rounded-lg bg-[#1D9E75] px-4 py-[7px] text-xs font-medium text-white no-underline hover:bg-[#0F6E56]"
            >
              {user ? "Start recording" : "Create an account"}
            </Link>
            <Link
              href="/login"
              className="rounded-lg border border-[#5DCAA5] bg-white px-4 py-[7px] text-xs font-medium text-[#1D9E75] no-underline"
            >
              Sign in with email
            </Link>
          </div>
        </div>
        <p className="mt-8 text-center text-[10px] text-[#94a3b8]">
          Email/password auth only · Raw audio is never client-encrypted · Results are AES-256-GCM
        </p>
      </main>
    </>
  );
}
