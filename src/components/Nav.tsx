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
      className={`text-sm no-underline ${pathname === href ? "text-[#1D9E75] font-medium" : "text-[#64748b] hover:text-[#0f172a]"}`}
    >
      {label}
    </Link>
  );

  return (
    <nav className="sticky top-0 z-10 border-b border-[#e2e8f0] bg-white">
      <div className="flex h-16 w-full items-center justify-between px-6 sm:px-8 lg:px-10">
        <Link href="/" className="flex items-center gap-2 no-underline">
          <span className="h-2.5 w-2.5 rounded-full bg-[#1D9E75]" />
          <span className="text-lg font-semibold tracking-tight text-[#0f172a]">MoodSense</span>
        </Link>
        <div className="hidden items-center gap-6 sm:flex">
          {email ? (
            <>
              {link("/dashboard", "History")}
              {link("/test", "Take test")}
              <button
                type="button"
                onClick={() => void signOut()}
                className="text-sm text-[#64748b] hover:text-[#0f172a]"
              >
                Sign out
              </button>
            </>
          ) : (
            <>
              {link("/login", "Sign in")}
              <Link
                href="/register"
                className="rounded-lg bg-[#1D9E75] px-4 py-2 text-xs font-medium text-white no-underline hover:bg-[#0F6E56]"
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
