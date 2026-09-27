"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { orderedEmotionCounts } from "@/lib/charts/aggregate";

const TEAL = "#1D9E75";

type EmotionBarProps = {
  counts: Record<string, number>;
};

export function EmotionBar({ counts }: EmotionBarProps) {
  const data = orderedEmotionCounts(counts);

  return (
    <section className="rounded-xl border border-[#e2e8f0] bg-white p-4">
      <h3 className="mb-3 text-sm font-semibold text-[#0f172a]">Emotion frequencies</h3>
      <div className="h-[280px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
            <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="emotion"
              tick={{ fill: "#475569", fontSize: 12 }}
              axisLine={{ stroke: "#94a3b8" }}
              tickLine={{ stroke: "#94a3b8" }}
            />
            <YAxis
              allowDecimals={false}
              tick={{ fill: "#475569", fontSize: 12 }}
              axisLine={{ stroke: "#94a3b8" }}
              tickLine={{ stroke: "#94a3b8" }}
              label={{
                value: "Frequency",
                angle: -90,
                position: "insideLeft",
                fill: "#475569",
                fontSize: 12,
              }}
            />
            <Tooltip
              cursor={{ fill: "#f8fafc" }}
              contentStyle={{
                background: "#ffffff",
                border: "1px solid #e2e8f0",
                borderRadius: 8,
                color: "#0f172a",
              }}
            />
            <Bar dataKey="count" name="Frequency" fill={TEAL} radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
