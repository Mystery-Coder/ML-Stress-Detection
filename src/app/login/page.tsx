import Link from "next/link";
import { AuthForm } from "@/components/AuthForm";
import { Nav } from "@/components/Nav";

export default function LoginPage() {
  return (
    <>
      <Nav />
      <main className="mx-auto w-full max-w-md px-8 py-16">
        <h1 className="font-[family-name:var(--font-display)] text-2xl text-[#0f172a]">Sign in</h1>
        <p className="mt-1 mb-6 text-sm">Email and password only. No OAuth, no magic links.</p>
        <AuthForm mode="login" />
        <p className="mt-4 text-xs">
          No account?{" "}
          <Link href="/register" className="text-[#1D9E75]">
            Register
          </Link>
        </p>
      </main>
    </>
  );
}
