import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  serverExternalPackages: ["onnxruntime-web"],
  turbopack: {},
};

export default nextConfig;
