"use client";

import { QUESTIONS } from "@/lib/questions";
import { Recorder } from "./Recorder";

type QuestionFlowProps = {
  index: number;
  onIndex: (index: number) => void;
  onBlob: (index: number, blob: Blob) => void;
};

export function QuestionFlow({ index, onIndex, onBlob }: QuestionFlowProps) {
  const last = index === QUESTIONS.length - 1;
  return (
    <div>
      <div className="mb-6 rounded-xl border border-[#e2e8f0] bg-white p-4">
        <p className="text-[10px] font-semibold uppercase tracking-[0.08em] text-[#94a3b8]">
          Question {index + 1} of {QUESTIONS.length}
        </p>
        <h2 className="mt-1 text-base font-medium leading-relaxed text-[#1e293b]">
          {QUESTIONS[index]}
        </h2>
      </div>
      <Recorder key={index} onBlob={(blob) => onBlob(index, blob)} />
      <div className="mt-6 flex gap-3">
        {!last ? (
          <button
            type="button"
            onClick={() => onIndex(index + 1)}
            className="flex-1 rounded-lg bg-[#1D9E75] px-4 py-[7px] text-xs font-medium text-white"
          >
            Next
          </button>
        ) : null}
        <button
          type="button"
          onClick={() => onIndex(QUESTIONS.length - 1)}
          className="rounded-lg border border-[#e2e8f0] bg-white px-4 py-[7px] text-xs font-medium text-[#64748b]"
        >
          Go to last
        </button>
      </div>
    </div>
  );
}
