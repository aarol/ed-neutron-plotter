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
    },
    proxy: {
      '/spansh-api': {
        target: 'https://spansh.co.uk/api',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/spansh-api/, '')
      }
    },
    watch: {
      ignored: [
        "rust-module/**/*",
      ]
    }
  },
  base: "/ed-neutron-plotter/"
};
