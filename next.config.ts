import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  outputFileTracingIncludes: {
    "/api/analyze": ["./models/**"],
  },
  serverExternalPackages: ["@tensorflow/tfjs-node", "@tensorflow/tfjs"],
};

export default nextConfig;
