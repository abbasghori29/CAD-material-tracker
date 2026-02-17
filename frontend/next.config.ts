import type { NextConfig } from "next";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

const nextConfig: NextConfig = {
  /* Disable React Strict Mode to prevent double-mount 
     which kills WebSocket connections in development */
  reactStrictMode: false,
  output: "standalone",

  experimental: {
    proxyClientMaxBodySize: "100mb",
  },

  /**
   * Proxy all backend API calls through Next.js.
   * Browser only talks to the Next.js server (same origin = no CORS).
   * Backend stays on localhost:8000 (not publicly exposed).
   */
  async rewrites() {
    return [
      // Auth routes (/auth/login, /auth/signup, /auth/me)
      { source: "/auth/:path*", destination: `${BACKEND_URL}/auth/:path*` },
      // Upload routes
      { source: "/upload", destination: `${BACKEND_URL}/upload` },
      { source: "/upload-tags", destination: `${BACKEND_URL}/upload-tags` },
      // Job routes
      { source: "/jobs", destination: `${BACKEND_URL}/jobs` },
      { source: "/jobs/:path*", destination: `${BACKEND_URL}/jobs/:path*` },
      // Download
      { source: "/download", destination: `${BACKEND_URL}/download` },
      { source: "/download-csv", destination: `${BACKEND_URL}/download-csv` },
      // Cleanup
      { source: "/cleanup", destination: `${BACKEND_URL}/cleanup` },
      // Static/output images served by backend
      { source: "/output_images/:path*", destination: `${BACKEND_URL}/output_images/:path*` },
      { source: "/static/:path*", destination: `${BACKEND_URL}/static/:path*` },
      // WebSocket
      { source: "/ws", destination: `${BACKEND_URL}/ws` },
    ];
  },
};

export default nextConfig;
