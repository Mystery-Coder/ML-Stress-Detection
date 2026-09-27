"use client";

import { useState } from "react";
import { DepressionLine } from "@/components/charts/DepressionLine";
import { EmotionBar } from "@/components/charts/EmotionBar";
import { EmotionLine } from "@/components/charts/EmotionLine";
import { EmotionPie } from "@/components/charts/EmotionPie";
import { countEmotions } from "@/lib/charts/aggregate";
import { unlockResults } from "@/lib/crypto/unlock";
import { createClient } from "@/lib/supabase/client";

type Predictions = { emotion: string[]; depression: string[] };

export function UnlockResults({ testId }: { testId: string }) {
  const [passcode, setPasscode] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<Predictions | null>(null);
  const [busy, setBusy] = useState(false);

  async function unlock() {
    setBusy(true);
    setError(null);
    const supabase = createClient();
    const {
      data: { user },
    } = await supabase.auth.getUser();
    await supabase.from("audit_log").insert({
      user_id: user?.id ?? null,
      action: "unlock_attempt",
      test_id: testId,
    });
    const { data: seal, error: sealError } = await supabase
      .from("result_seals")
      .select("ciphertext, iv, salt")
      .eq("test_id", testId)
      .maybeSingle();
    if (sealError || !seal) {
      setBusy(false);
      setError("No sealed result found for this test.");
      return;
    }
    try {
      const payload = await unlockResults<Predictions>(passcode, seal);
      setData(payload);
      await supabase.from("audit_log").insert({
        user_id: user?.id ?? null,
        action: "unlock_success",
        test_id: testId,
      });
    } catch {
      setError("Invalid key — access denied");
      await supabase.from("audit_log").insert({
        user_id: user?.id ?? null,
        action: "unlock_denied",
        test_id: testId,
      });
    } finally {
      setBusy(false);
    }
  }

  if (!data) {
    return (
      <div className="mx-auto max-w-md rounded-xl border border-[#e2e8f0] bg-white p-6">
        <h1 className="font-[family-name:var(--font-display)] text-2xl text-[#0f172a]">
          Enter your key to unlock your results
        </h1>
        <p className="mt-2 text-sm">Wrong key or a tampered seal fails closed. The server never stores this passcode.</p>
        <input
          type="password"
          value={passcode}
          onChange={(e) => setPasscode(e.target.value)}
          className="mt-4 w-full rounded-lg border border-[#e2e8f0] px-3 py-2 text-sm text-[#0f172a] outline-none focus:border-[#1D9E75]"
        />
        {error ? <p className="mt-2 text-xs text-[#E24B4A]">{error}</p> : null}
        <button
          type="button"
          disabled={busy || !passcode}
          onClick={() => void unlock()}
          className="mt-4 w-full rounded-lg bg-[#1D9E75] px-4 py-2 text-xs font-medium text-white disabled:opacity-40"
        >
          Unlock
        </button>
      </div>
    );
  }

  const counts = countEmotions(data.emotion);
  return (
    <div>
      <h1 className="font-[family-name:var(--font-display)] text-2xl text-[#0f172a]">Your emotional profile</h1>
      <p className="mb-6 text-xs text-[#94a3b8]">Sealed client-side · MFCC · 3 s emotion ticks · 2 min depression ticks</p>
      <div className="mb-8 grid grid-cols-2 gap-2 sm:grid-cols-4">
        {Object.entries(counts).map(([emotion, count]) => (
          <div key={emotion} className="rounded-lg border border-[#e2e8f0] bg-[#f8fafc] p-2 text-center">
            <span className="block text-[9px] capitalize text-[#94a3b8]">{emotion}</span>
            <div className="text-base font-semibold text-[#0f172a]">{count}</div>
          </div>
        ))}
      </div>
      <div className="space-y-6">
        <EmotionBar counts={counts} />
        <EmotionLine emotions={data.emotion} />
        <DepressionLine depression={data.depression} />
        <EmotionPie counts={counts} />
      </div>
    </div>
  );
}
