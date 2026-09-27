"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { EMOTION_Y_ORDER, type EmotionLabel } from "@/lib/charts/aggregate";

const TEAL = "#1D9E75";

type EmotionLineProps = {
  emotions: string[];
};

export function EmotionLine({ emotions }: EmotionLineProps) {
  const data = emotions.flatMap((emotion, index) => {
    const value = EMOTION_Y_ORDER.indexOf(emotion as EmotionLabel);
    if (value < 0) return [];
    return [{ time: index * 3, value, emotion }];
  });

  return (
    <section className="rounded-xl border border-[#e2e8f0] bg-white p-4">
      <h3 className="mb-3 text-sm font-semibold text-[#0f172a]">
        Emotion transition over time
      </h3>
      <div className="h-[280px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 12, left: 8, bottom: 8 }}>
            <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" />
            <XAxis
              dataKey="time"
              type="number"
              domain={[0, "dataMax"]}
              tick={{ fill: "#475569", fontSize: 12 }}
              axisLine={{ stroke: "#94a3b8" }}
              tickLine={{ stroke: "#94a3b8" }}
              label={{
                value: "Time (seconds)",
                position: "insideBottom",
                offset: -2,
                fill: "#475569",
                fontSize: 12,
              }}
            />
            <YAxis
              type="number"
              domain={[-0.5, EMOTION_Y_ORDER.length - 0.5]}
              ticks={EMOTION_Y_ORDER.map((_, index) => index)}
              tickFormatter={(value: number) => EMOTION_Y_ORDER[value] ?? ""}
              width={78}
              tick={{ fill: "#475569", fontSize: 12 }}
              axisLine={{ stroke: "#94a3b8" }}
              tickLine={{ stroke: "#94a3b8" }}
            />
            <Tooltip
              contentStyle={{
                background: "#ffffff",
                border: "1px solid #e2e8f0",
                borderRadius: 8,
                color: "#0f172a",
              }}
              labelFormatter={(label) => `${label}s`}
              formatter={(value) => [
                EMOTION_Y_ORDER[Number(value)] ?? String(value),
                "Emotion",
              ]}
            />
            <Line
              type="linear"
              dataKey="value"
              name="Emotion path"
              stroke={TEAL}
              strokeWidth={2}
              dot={{ r: 3, fill: TEAL, stroke: TEAL }}
              activeDot={{ r: 5 }}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
