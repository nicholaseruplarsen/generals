/// <reference types="vitest/config" />
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

// Deployed at https://nicholaseruplarsen.github.io/generals/ — base must match.
export default defineConfig({
  base: "/generals/",
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
