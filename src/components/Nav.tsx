"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { createClient } from "@/lib/supabase/client";

export function Nav({ email }: { email?: string | null }) {
  const pathname = usePathname();
  const router = useRouter();

  async function signOut() {
    const supabase = createClient();
    await supabase.auth.signOut();
    router.push("/");
    router.refresh();
  }

  const link = (href: string, label: string) => (
    <Link
      href={href}
      className={`text-xs no-underline ${pathname === href ? "text-[#1D9E75] font-medium" : "text-[#64748b]"}`}
    >
      {label}
    </Link>
  );

  return (
    <nav className="sticky top-0 z-10 border-b border-[#e2e8f0] bg-white">
      <div className="mx-auto flex h-14 max-w-4xl items-center justify-between px-8">
        <Link href="/" className="flex items-center gap-1.5 no-underline">
          <span className="h-2 w-2 rounded-full bg-[#1D9E75]" />
          <span className="text-[13px] font-semibold text-[#0f172a]">MoodSense</span>
        </Link>
        <div className="hidden items-center gap-4 sm:flex">
          {email ? (
            <>
              {link("/dashboard", "History")}
              {link("/test", "Take test")}
              <button
                type="button"
                onClick={() => void signOut()}
                className="text-xs text-[#64748b]"
              >
                Sign out
              </button>
            </>
          ) : (
            <>
              {link("/login", "Sign in")}
              <Link
                href="/register"
                className="rounded-lg bg-[#1D9E75] px-3 py-[5px] text-[11px] font-medium text-white no-underline hover:bg-[#0F6E56]"
              >
                Create account
              </Link>
            </>
          )}
        </div>
      </div>
    </nav>
  );
}
