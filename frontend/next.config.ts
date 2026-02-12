import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* Disable React Strict Mode to prevent double-mount 
     which kills WebSocket connections in development */
  reactStrictMode: false,
  output: "standalone",
};

export default nextConfig;
