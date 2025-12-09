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

async function getStarCoords(star: string): Promise<Vector3> {
  const apiUrl = "https://www.edsm.net/api-v1/system"

  const response = await fetch(`${apiUrl}?systemName=${encodeURIComponent(star)}&showCoordinates=1`);

  if (!response.ok) {
    throw new Error(`Error fetching star coordinates: ${response.statusText}`);
  }

  const data: SystemInfoResponse = await response.json();

  return new Vector3(
    -data.coords.x, // All x-coordinates are negative to what you see in game
    data.coords.y,
    data.coords.z
  ).divideScalar(1000)
}

export const api = {
  getStarCoords,
}
