"use client";

import { useEffect, useRef, useState } from "react";

type RecorderProps = {
  onBlob: (blob: Blob) => void;
};

export function Recorder({ onBlob }: RecorderProps) {
  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const chunks = useRef<Blob[]>([]);
  const recorder = useRef<MediaRecorder | null>(null);
  const stream = useRef<MediaStream | null>(null);
  const timer = useRef<number | null>(null);
  const lastDuration = useRef(0);
  const audioEl = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    return () => {
      if (timer.current) window.clearInterval(timer.current);
      stream.current?.getTracks().forEach((track) => track.stop());
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
  }, [audioUrl]);

  async function start() {
    chunks.current = [];
    setProgress(0);
    setSeconds(0);
    stream.current = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
      ? "audio/webm;codecs=opus"
      : "audio/webm";
    const rec = new MediaRecorder(stream.current, {
      mimeType,
      audioBitsPerSecond: 128000,
    });
    rec.ondataavailable = (event) => {
      if (event.data.size > 0) chunks.current.push(event.data);
    };
    rec.onstop = () => {
      const blob = new Blob(chunks.current, { type: mimeType });
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      const url = URL.createObjectURL(blob);
      setAudioUrl(url);
      lastDuration.current = seconds;
      onBlob(blob);
      stream.current?.getTracks().forEach((track) => track.stop());
      setRecording(false);
    };
    recorder.current = rec;
    rec.start(1000);
    setRecording(true);
    timer.current = window.setInterval(() => setSeconds((s) => s + 1), 1000);
  }

  function stop() {
    if (timer.current) window.clearInterval(timer.current);
    recorder.current?.stop();
  }

  function play() {
    if (!audioUrl) return;
    const audio = new Audio(audioUrl);
    audioEl.current = audio;
    setPlaying(true);
    const id = window.setInterval(() => {
      if (lastDuration.current > 0) {
        setProgress(Math.min(100, (audio.currentTime / lastDuration.current) * 100));
      }
    }, 100);
    audio.onended = () => {
      window.clearInterval(id);
      setPlaying(false);
      setProgress(100);
    };
    void audio.play();
  }

  function stopPlay() {
    audioEl.current?.pause();
    setPlaying(false);
  }

  const mm = Math.floor(seconds / 60);
  const ss = String(seconds % 60).padStart(2, "0");

  return (
    <div>
      <div className="mb-6 flex flex-col items-center rounded-xl border border-[#e2e8f0] bg-white p-6">
        <div
          className={`mb-3 flex h-14 w-14 items-center justify-center rounded-full border-2 ${recording ? "border-[#E24B4A]" : "border-[#1D9E75]"}`}
        >
          <div
            className={`flex h-10 w-10 items-center justify-center rounded-full ${recording ? "bg-[#E24B4A]" : "bg-[#1D9E75]"}`}
          >
            <svg className="h-[18px] w-[18px] fill-none stroke-white stroke-2" viewBox="0 0 24 24">
              <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
              <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
            </svg>
          </div>
        </div>
        <p className="text-lg font-semibold tabular-nums text-[#1D9E75]">
          {mm}:{ss}
        </p>
        <p className="mt-1 text-[11px] text-[#64748b]">Speak clearly for best results</p>
      </div>
      {recording ? (
        <div className="mb-4 flex items-center justify-center gap-3 rounded-lg border border-[#F09595] bg-[#FCEBEB] p-3">
          <span className="h-3 w-3 animate-pulse rounded-full bg-[#E24B4A]" />
          <span className="text-sm font-medium text-[#A32D2D]">Recording in progress…</span>
        </div>
      ) : null}
      <div className="mb-4 flex flex-wrap justify-center gap-3">
        <button
          type="button"
          disabled={recording}
          onClick={() => void start()}
          className="rounded-lg bg-[#1D9E75] px-4 py-[7px] text-xs font-medium text-white disabled:opacity-40"
        >
          Start recording
        </button>
        <button
          type="button"
          disabled={!recording}
          onClick={stop}
          className="rounded-lg bg-[#E24B4A] px-4 py-[7px] text-xs font-medium text-white disabled:opacity-40"
        >
          Stop recording
        </button>
        <button
          type="button"
          disabled={!audioUrl || recording}
          onClick={play}
          className="rounded-lg border border-[#5DCAA5] bg-white px-4 py-[7px] text-xs font-medium text-[#1D9E75] disabled:opacity-40"
        >
          Play
        </button>
        <button
          type="button"
          disabled={!playing}
          onClick={stopPlay}
          className="rounded-lg border border-[#e2e8f0] bg-white px-4 py-[7px] text-xs font-medium text-[#64748b] disabled:opacity-40"
        >
          Stop audio
        </button>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-[#e2e8f0]">
        <div className="h-full rounded-full bg-[#1D9E75]" style={{ width: `${progress}%` }} />
      </div>
    </div>
  );
}
