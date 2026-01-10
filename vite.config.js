import wasm from "vite-plugin-wasm";

/** @type {import('vite').UserConfig} */
export default {
  plugins: [wasm()],
  server: {
    allowedHosts: true,
    headers: {
      "Cross-Origin-Embedder-Policy": "require-corp",
      "Cross-Origin-Opener-Policy": "same-origin"
    }
  },
  base: "/ed-neutron-plotter/"
};
