# Elite Dangerous neutron plotter

A website which shows a visualization of the known neutron stars in  Elite Dangerous, and allows one to find the path with the least jumps from point A to B, using neutron stars.

## Features

- Star visualizer rendering 4.4 million stars at 60/144 FPS via Three.js and WebGPU
- Extremely overengineered local autocomplete system for the search box
  - Compresses 4.4 million stars into <17 MB gzipped and allows instant autocomplete via a very fast Rust WebAssembly module (see [rust-module/](./rust-module/))
- Plotter for finding the fastest route between stars (WIP)

## Development

The web UI is written in Typescript, using Three.js. The pathfinding and autocomplete system uses Rust in a WebAssembly module ([rust-module](./rust-module/)).

Node.js, pnpm and cargo need to be installed.

See [rust-module/README.md](./rust-module/README.md) for generating the data.

After the data is generated in the "./public/data" folder, run the dev server with ```pnpm dev```.
