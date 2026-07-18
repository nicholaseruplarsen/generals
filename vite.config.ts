/// <reference types="vitest/config" />
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

// The site (Hugo blog, blog/) deploys at /generals/; the game lives under it
// at /generals/play/. `npm run site` builds both into _site/generals/.
export default defineConfig({
  base: "/generals/play/",
  build: {
    outDir: "_site/generals/play",
  },
  resolve: {
    alias: [
      // onnxruntime-web's exports map blocks deep imports; the worker needs
      // the raw wasm binary as a Vite asset (?url) to hand to ort at runtime.
      // Regex form: string aliases match exactly and would miss the ?url query.
      {
        find: /^@ort-wasm/,
        replacement: fileURLToPath(new URL(
          "./node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm",
          import.meta.url,
        )),
      },
    ],
  },
  test: {
    environment: "node",
    include: ["test/**/*.test.ts"],
  },
});
