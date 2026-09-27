"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { mergeBlobsToWav } from "@/lib/audio/merge";
import { sealResults } from "@/lib/crypto/seal";
import { QUESTIONS } from "@/lib/questions";
import { createClient } from "@/lib/supabase/client";
import { QuestionFlow } from "./QuestionFlow";

type Predictions = { emotion: string[]; depression: string[] };

export function TestSession() {
  const router = useRouter();
  const [passcode, setPasscode] = useState("");
  const [started, setStarted] = useState(false);
  const [index, setIndex] = useState(0);
  const [blobs, setBlobs] = useState<Map<number, Blob>>(new Map());
  const [status, setStatus] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const answered = useMemo(() => blobs.size, [blobs]);

  function storeBlob(questionIndex: number, blob: Blob) {
    setBlobs((prev) => {
      const next = new Map(prev);
      next.set(questionIndex, blob);
      return next;
    });
  }

  async function submit() {
    if (passcode.length < 8) {
      setStatus("Passcode must be at least 8 characters.");
      return;
    }
    const ordered = QUESTIONS.map((_, i) => blobs.get(i)).filter((b): b is Blob => Boolean(b));
    if (ordered.length === 0) {
      setStatus("Record at least one answer before submitting.");
      return;
    }
    setBusy(true);
    setStatus("Merging recordings…");
    try {
      const supabase = createClient();
      const {
        data: { user },
      } = await supabase.auth.getUser();
      if (!user) {
        setStatus("You need to sign in.");
        return;
      }

      const { data: test, error: testError } = await supabase
        .from("tests")
        .insert({ user_id: user.id, status: "recording" })
        .select("id")
        .single();
      if (testError || !test) {
        setStatus(testError?.message ?? "Could not create test");
        return;
      }

      const wav = await mergeBlobsToWav(ordered, (done, total) => {
        setStatus(`Merging recordings… ${done}/${total}`);
      });
      const audioPath = `${user.id}/${test.id}/merged.wav`;
      setStatus("Uploading audio…");
      const { error: uploadError } = await supabase.storage.from("audio").upload(audioPath, wav, {
        contentType: "audio/wav",
        upsert: true,
      });
      if (uploadError) {
        setStatus(uploadError.message);
        return;
      }
      await supabase
        .from("tests")
        .update({ audio_path: audioPath, status: "processing" })
        .eq("id", test.id);

      setStatus("Analyzing on the server…");
      const analyze = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ testId: test.id }),
      });
      const payload = (await analyze.json()) as { predictions?: Predictions; error?: string };
      if (!analyze.ok || !payload.predictions) {
        await supabase.from("tests").update({ status: "error" }).eq("id", test.id);
        setStatus(payload.error ?? "Analysis failed");
        return;
      }

      setStatus("Sealing results with your key…");
      const seal = await sealResults(passcode, payload.predictions);
      const { error: sealError } = await supabase.from("result_seals").upsert({
        test_id: test.id,
        ...seal,
      });
      if (sealError) {
        setStatus(sealError.message);
        return;
      }
      await supabase
        .from("tests")
        .update({ status: "completed", completed_at: new Date().toISOString() })
        .eq("id", test.id);
      router.push(`/result/${test.id}`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Submit failed");
    } finally {
      setBusy(false);
    }
  }

  if (!started) {
    return (
      <div className="rounded-xl border border-[#e2e8f0] bg-white p-6">
        <h1 className="font-[family-name:var(--font-display)] text-2xl text-[#0f172a]">
          Set a results key
        </h1>
        <p className="mt-2 text-sm leading-relaxed">
          This passcode never leaves your browser. It derives an AES-256-GCM key used to seal
          predictions. If you lose it, the stored results cannot be recovered.
        </p>
        <input
          type="password"
          minLength={8}
          value={passcode}
          onChange={(e) => setPasscode(e.target.value)}
          placeholder="At least 8 characters"
          className="mt-4 w-full rounded-lg border border-[#e2e8f0] px-3 py-2 text-sm text-[#0f172a] outline-none focus:border-[#1D9E75]"
        />
        <button
          type="button"
          onClick={() => {
            if (passcode.length < 8) return;
            setStarted(true);
          }}
          className="mt-4 rounded-lg bg-[#1D9E75] px-4 py-2 text-xs font-medium text-white hover:bg-[#0F6E56]"
        >
          Begin recording
        </button>
      </div>
    );
  }

  return (
    <div>
      <p className="mb-4 text-[11px] text-[#94a3b8]">
        {answered} of {QUESTIONS.length} answers recorded
      </p>
      <QuestionFlow index={index} onIndex={setIndex} onBlob={storeBlob} />
      {index === QUESTIONS.length - 1 ? (
        <button
          type="button"
          disabled={busy}
          onClick={() => void submit()}
          className="mt-6 w-full rounded-lg bg-[#1D9E75] px-4 py-2 text-xs font-medium text-white disabled:opacity-40"
        >
          {busy ? status ?? "Working…" : "Submit and analyze"}
        </button>
      ) : null}
      {status && index !== QUESTIONS.length - 1 ? (
        <p className="mt-3 text-xs text-[#E24B4A]">{status}</p>
      ) : null}
      {busy ? <p className="mt-3 text-xs text-[#0F6E56]">{status}</p> : null}
    </div>
  );
}
