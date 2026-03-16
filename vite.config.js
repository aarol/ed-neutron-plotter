import wasm from "vite-plugin-wasm";
import preact from "@preact/preset-vite";
import tailwindcss from "@tailwindcss/vite";

/** @type {import('vite').UserConfig} */
export default {
  plugins: [wasm(), preact(), tailwindcss()],
  server: {
    allowedHosts: true,
    headers: {
      "Cross-Origin-Embedder-Policy": "require-corp",
      "Cross-Origin-Opener-Policy": "same-origin"
    }
  },
  base: "/ed-neutron-plotter/"
};
