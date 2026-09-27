"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { createClient } from "@/lib/supabase/client";

type Mode = "login" | "register";

export function AuthForm({ mode }: { mode: Mode }) {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    setInfo(null);
    const supabase = createClient();
    if (mode === "register") {
      const { error: signUpError } = await supabase.auth.signUp({ email, password });
      setBusy(false);
      if (signUpError) {
        setError(signUpError.message);
        return;
      }
      setInfo("Account created. Confirm your email if required, then sign in.");
      router.push("/login");
      router.refresh();
      return;
    }
    const { error: signInError } = await supabase.auth.signInWithPassword({ email, password });
    setBusy(false);
    if (signInError) {
      setError(signInError.message);
      return;
    }
    router.push("/dashboard");
    router.refresh();
  }

  return (
    <form onSubmit={(e) => void onSubmit(e)} className="space-y-4">
      <label className="block text-xs font-medium text-[#334155]">
        Email
        <input
          type="email"
          required
          autoComplete="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          className="mt-1 w-full rounded-lg border border-[#e2e8f0] px-3 py-2 text-sm text-[#0f172a] outline-none focus:border-[#1D9E75]"
        />
      </label>
      <label className="block text-xs font-medium text-[#334155]">
        Password
        <input
          type="password"
          required
          minLength={6}
          autoComplete={mode === "login" ? "current-password" : "new-password"}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="mt-1 w-full rounded-lg border border-[#e2e8f0] px-3 py-2 text-sm text-[#0f172a] outline-none focus:border-[#1D9E75]"
        />
      </label>
      {error ? <p className="text-xs text-[#E24B4A]">{error}</p> : null}
      {info ? <p className="text-xs text-[#0F6E56]">{info}</p> : null}
      <button
        type="submit"
        disabled={busy}
        className="w-full rounded-lg bg-[#1D9E75] px-4 py-2 text-xs font-medium text-white hover:bg-[#0F6E56] disabled:opacity-40"
      >
        {mode === "login" ? "Sign in" : "Create account"}
      </button>
    </form>
  );
}
