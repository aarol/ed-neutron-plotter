import { Vector3 } from "three";

export interface SystemInfoResponse {
  name: string
  coords: Coords
}

export interface Coords {
  x: number
  y: number
  z: number
}

export function vec3FromCoords(coords: Coords): Vector3 {
  return new Vector3(coords.x, coords.y, coords.z);
}

async function getStarCoords(star: string): Promise<Coords> {
  const apiUrl = "https://www.edsm.net/api-v1/system"

  const response = await fetch(`${apiUrl}?systemName=${encodeURI(star)}&showCoordinates=1`);

  if (!response.ok) {
    throw new Error(`Error fetching star coordinates: ${response.statusText}`);
  }

  const data: SystemInfoResponse = await response.json();

  return {
    x: -data.coords.x / 1000,
    y: data.coords.y / 1000,
    z: data.coords.z / 1000,
  }
}

export const api = {
  getStarCoords,
}
