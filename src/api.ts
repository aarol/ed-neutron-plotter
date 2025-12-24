import { Vector3 } from "three";
import * as wasm from "../rust-module/pkg";

export const wasmModule = new wasm.Module();

export type SystemInfoResponse = {
  name: string
  coords: ApiCoords
} | []

export interface ApiCoords {
  x: number
  y: number
  z: number
}

export function vec3fromCoords(coords: ApiCoords): Vector3 {
  return new Vector3(coords.x, coords.y, coords.z);
}

function wasmCoordsToCoordsObj(coords: wasm.Coords): ApiCoords {
  return {
    x: coords.x,
    y: coords.y,
    z: coords.z,
  }
}

async function getStarCoords(star: string): Promise<ApiCoords | null> {
  // First try to get coordinates from the wasm module's searcher
  let wasmCoords = wasmModule.get_coords_for_star(star);
  if (wasmCoords) {
    return wasmCoordsToCoordsObj(wasmCoords);
  }

  // If not found, fall back to the API
  return await getStarCoordsFromApi(star);
}

async function getStarCoordsFromApi(star: string): Promise<ApiCoords | null> {
  const apiUrl = "https://www.edsm.net/api-v1/system"

  const response = await fetch(`${apiUrl}?systemName=${encodeURI(star)}&showCoordinates=1`);

  if (!response.ok) {
    throw new Error(`Error fetching star coordinates: ${response.statusText}`);
  }

  const data: SystemInfoResponse = await response.json();
  if (Array.isArray(data)) {
    // If the star is not found, EDSM returns an empty array
    return null;
  }

  return {
    x: -data.coords.x / 1000,
    y: data.coords.y / 1000,
    z: data.coords.z / 1000,
  }
}

function findRoute(start: ApiCoords, end: ApiCoords, reportCallback: (report: any) => void): ApiCoords[] | undefined {
  const wasmStart = new wasm.Coords(start.x, start.y, start.z);
  const wasmEnd = new wasm.Coords(end.x, end.y, end.z);

  const result = wasmModule.find_route(wasmStart, wasmEnd, reportCallback);
  return result?.map(wasmCoordsToCoordsObj);
}

export const api = {
  getStarCoords,
  findRoute,
}
