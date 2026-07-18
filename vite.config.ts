/// <reference types="vitest/config" />
import { defineConfig } from "vite";

// Deployed at https://nicholaseruplarsen.github.io/generals/ — base must match.
export default defineConfig({
  base: "/generals/",
  test: {
    environment: "node",
    include: ["test/**/*.test.ts"],
  },
});
