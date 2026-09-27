"use client";

import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import { orderedEmotionCounts } from "@/lib/charts/aggregate";

const SLICE_COLORS = [
  "#1D9E75",
  "#0f766e",
  "#0f172a",
  "#334155",
  "#475569",
  "#64748b",
  "#94a3b8",
  "#cbd5e1",
];

type EmotionPieProps = {
  counts: Record<string, number>;
};

export function EmotionPie({ counts }: EmotionPieProps) {
  const data = orderedEmotionCounts(counts);
  const total = data.reduce((sum, row) => sum + row.count, 0);

  return (
    <section className="rounded-xl border border-[#e2e8f0] bg-white p-4">
      <h3 className="mb-3 text-sm font-semibold text-[#0f172a]">Emotion distribution</h3>
      <div className="h-[280px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              dataKey="count"
              nameKey="emotion"
              cx="50%"
              cy="50%"
              innerRadius={48}
              outerRadius={90}
              paddingAngle={2}
              stroke="#ffffff"
            >
              {data.map((row, index) => (
                <Cell key={row.emotion} fill={SLICE_COLORS[index % SLICE_COLORS.length]} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{
                background: "#ffffff",
                border: "1px solid #e2e8f0",
                borderRadius: 8,
                color: "#0f172a",
              }}
              formatter={(value, name) => {
                const count = Number(value);
                const pct = total > 0 ? ((count / total) * 100).toFixed(1) : "0.0";
                return [`${count} (${pct}%)`, name];
              }}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
