
type SpanshSystemResponse = {
  name: string;
  x: number;
  y: number;
  z: number;
}

export async function getSystemCoords(id64: number): Promise<{ x: number; y: number; z: number } | null> {
  const response = await fetch(`/spansh-api/system/${id64}`);
  if (!response.ok) {
    console.error(`Error fetching Spansh system data: ${response.statusText}`);
    return null;
  }

  const data: SpanshSystemResponse = await response.json();
  return {
    x: data.x,
    y: data.y,
    z: data.z
  };
}

export default {
  getSystemCoords,
}