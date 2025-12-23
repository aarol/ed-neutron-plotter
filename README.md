# Elite Dangerous neutron plotter

A website which shows a visualization of the known neutron stars in  Elite Dangerous, and allows one to find the path with the least jumps from point A to B, using neutron stars.

## Features

- Star visualizer rendering 4.4 million stars at 60/144 FPS via Three.js and WebGPU
- Extremely overengineered local autocomplete system for the search box
  - Compresses 4.4 million stars into <17 MB gzipped and allows instant autocomplete via a very fast Rust WebAssembly module (see [rust-module/](./rust-module/))
- Plotter for finding the fastest route between stars (WIP)
