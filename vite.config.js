import wasm from "vite-plugin-wasm";
import preact from "@preact/preset-vite";

/** @type {import('vite').UserConfig} */
export default {
  plugins: [wasm(), preact()],
  server: {
    allowedHosts: true,
    headers: {
      "Cross-Origin-Embedder-Policy": "require-corp",
      "Cross-Origin-Opener-Policy": "same-origin"
    }
  },
  base: "/ed-neutron-plotter/"
};
