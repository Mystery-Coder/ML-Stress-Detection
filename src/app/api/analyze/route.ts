import { NextRequest, NextResponse } from "next/server";
import { createServiceClient } from "@/lib/supabase/admin";
import { createClient } from "@/lib/supabase/server";

export const runtime = "nodejs";
export const maxDuration = 60;

function inferenceUrl() {
  return (process.env.INFERENCE_URL ?? "http://127.0.0.1:8001").replace(/\/$/, "");
}

export async function POST(request: NextRequest) {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = (await request.json()) as { testId?: string };
  if (!body.testId) {
    return NextResponse.json({ error: "testId required" }, { status: 400 });
  }

  const { data: test, error: testError } = await supabase
    .from("tests")
    .select("id, user_id, audio_path")
    .eq("id", body.testId)
    .maybeSingle();
  if (testError || !test || test.user_id !== user.id || !test.audio_path) {
    return NextResponse.json({ error: "Test not found" }, { status: 404 });
  }

  const admin = createServiceClient();
  const { data: file, error: downloadError } = await admin.storage.from("audio").download(test.audio_path);
  if (downloadError || !file) {
    return NextResponse.json({ error: downloadError?.message ?? "Audio missing" }, { status: 404 });
  }

  const wav = await file.arrayBuffer();
  let worker: Response;
  try {
    worker = await fetch(`${inferenceUrl()}/predict`, {
      method: "POST",
      headers: { "Content-Type": "audio/wav" },
      body: wav,
    });
  } catch {
    return NextResponse.json(
      { error: "Inference worker is not running. Start it with: python -m infer.app" },
      { status: 503 },
    );
  }

  const payload = (await worker.json()) as {
    predictions?: { emotion: string[]; depression: string[] };
    detail?: string;
  };
  if (!worker.ok || !payload.predictions) {
    return NextResponse.json(
      { error: payload.detail ?? "Inference worker failed" },
      { status: worker.status === 400 ? 400 : 502 },
    );
  }

  return NextResponse.json({ predictions: payload.predictions });
}
