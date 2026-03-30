import { Vector3 } from "three";
import * as wasm from "../rust-module/pkg";
import type { StarSystem } from "./ui/types";

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

async function getStarCoords(wasmModule: wasm.Module, star: string): Promise<ApiCoords | null> {
  // First try to get coordinates from the wasm module's searcher
  // If not found, fall back to the API
  return wasmModule.get_coords_for_star(star) ?? await getStarCoordsFromApi(star);
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

async function getMultipleStarCoordsFromApi(stars: string[]): Promise<Record<string, ApiCoords>> {
  const apiUrl = "https://www.edsm.net/api-v1/systems"

  const url = `${apiUrl}?${encodeSystemNamesForApi(stars)}&showCoordinates=1`;
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Error fetching star coordinates: ${response.statusText}`);
  }
  const data = await response.json() as StarSystem[];
  
  const map = data.reduce((acc, system) => {
    acc[system.name] = {
      x: -system.coords.x / 1000,
      y: system.coords.y / 1000,
      z: system.coords.z / 1000,
    };
    return acc;
  }, {} as Record<string, ApiCoords>);

  for(const star of stars) {
    if (!map[star]) {
      console.log(`Star not found in API response: ${star}. Encoded URL: ${encodeURI(star)}`);
    }
  }


  return map;
}

function encodeSystemNamesForApi(stars: string[]): string {
  return stars.map(star => `systemName[]=${encodeURIComponent(star)}`).join("&");
}

export const api = {
  getStarCoords,
  getMultipleStarCoordsFromApi,
}
