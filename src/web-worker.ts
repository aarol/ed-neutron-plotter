import * as Comlink from "comlink";
import init, { Module } from "../rust-module/pkg/rust_module";
import type { ApiCoords } from "./api";
// import { api } from "./api";

export class WasmWorker {
  wasmModule!: Module;

  constructor() {
    init().then(() => {
      this.wasmModule = new Module();
    })
  }

  findRoute(start: ApiCoords, end: ApiCoords, reportCallback: (starData: Float32Array, distance: number, depth: number) => void): Float32Array| undefined {
    return this.wasmModule.find_route(start, end, (starData: Float32Array, distance: number, depth: number) => {
      reportCallback(starData, distance, depth);
    });
  }

  setStars(stars: Float32Array) {
    this.wasmModule.set_stars(stars);
    console.log("Wasm worker: Stars data set.");
  }

  setKDTree(kdtree: Uint8Array) {
    this.wasmModule.set_kdtree(kdtree);
    console.log("Wasm worker: KDTree data set.");
  }
}


Comlink.expose(WasmWorker)