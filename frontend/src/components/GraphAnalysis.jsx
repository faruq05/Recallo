import React from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

// Map score to color
const getDotColor = (score) => (score <= 7 ? "#ff4d4d" : "#28a745");

const GraphAnalysis = ({ topicTitle, history }) => {
  // Prepare chart data
  const chartData = history
    .slice()
    .reverse()
    .map((attempt, index) => ({
      attempt: `#${index + 1}`,
      score: attempt.score,
      date: new Date(attempt.timestamp || attempt.created_at).toLocaleDateString(),
    }));

  return (
    <div className="mb-4">
      <h5 className="text-white text-center mb-3">📈 Score Progress for: {topicTitle}</h5>
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="colorScore" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#00c9ff" stopOpacity={0.8} />
              <stop offset="100%" stopColor="#92fe9d" stopOpacity={0.2} />
            </linearGradient>
          </defs>

          <CartesianGrid strokeDasharray="3 3" stroke="#555" />
          <XAxis dataKey="attempt" stroke="#aaa" />
          <YAxis domain={[0, 10]} tickCount={6} stroke="#aaa" />
          <Tooltip
            contentStyle={{ backgroundColor: "#222", border: "none", color: "#fff" }}
            labelStyle={{ color: "#fff" }}
            formatter={(value, name) =>
              name === "score" ? [`${value}/10`, "Score"] : value
            }
          />
          <Area
            type="monotone"
            dataKey="score"
            stroke="#ffffff"
            fill="url(#colorScore)"
            activeDot={({ payload }) => ({
              r: 6,
              fill: getDotColor(payload.score),
              stroke: "#fff",
              strokeWidth: 2,
            })}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};

export default GraphAnalysis;
