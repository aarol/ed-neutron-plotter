import * as wasm from "../../rust-module/pkg/rust_module";
import eddb from "./eddb";

export interface ApiCoords {
  x: number
  y: number
  z: number
}

async function getStarCoords(wasmModule: wasm.Module, star: string): Promise<ApiCoords | null> {
  // First try to get coordinates from the wasm module's searcher
  // If not found, fall back to the API
  return wasmModule.get_coords_for_star(star) ?? await eddb.getStarCoordsFromApi(star);
}

export const api = {
  getStarCoords,
}