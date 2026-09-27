import Link from "next/link";
import { AuthForm } from "@/components/AuthForm";
import { Nav } from "@/components/Nav";

export default function RegisterPage() {
  return (
    <>
      <Nav />
      <main className="mx-auto w-full max-w-md px-8 py-16">
        <h1 className="font-[family-name:var(--font-display)] text-2xl text-[#0f172a]">
          Create account
        </h1>
        <p className="mt-1 mb-6 text-sm">Use any email. Confirmation depends on your Supabase project settings.</p>
        <AuthForm mode="register" />
        <p className="mt-4 text-xs">
          Already registered?{" "}
          <Link href="/login" className="text-[#1D9E75]">
            Sign in
          </Link>
        </p>
      </main>
    </>
  );
}
