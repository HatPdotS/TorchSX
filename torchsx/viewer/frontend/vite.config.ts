import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `base: "./"` makes the built bundle use relative asset paths, so FastAPI can
// serve dist/ from "/" unchanged. In dev, /api is proxied to the Python server
// (default port 8080); override with VITE_API_PORT.
const apiPort = process.env.VITE_API_PORT ?? "8080";

export default defineConfig({
  base: "./",
  plugins: [react()],
  server: {
    proxy: {
      "/api": { target: `http://localhost:${apiPort}`, changeOrigin: true },
    },
  },
});
