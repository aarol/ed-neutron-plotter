import { z } from "zod";


const storedRoutePlotSchema = z.object({
  nodes: z.array(z.object({
    name: z.string(),
    coords: z.object({
      x: z.number(),
      y: z.number(),
      z: z.number(),
    }),
  })),
  progress: z.number().int().min(0),
});

export type StoredRoutePlot = z.infer<typeof storedRoutePlotSchema>;

const STORAGE_KEY = "ed-galaxy.route-plot";

export function loadStoredRoutePlot(): StoredRoutePlot | null {
  const rawValue = localStorage.getItem(STORAGE_KEY);
  if (!rawValue) {
    return null;
  }

  try {
    const parsedValue = JSON.parse(rawValue);
    const parsedRoute = storedRoutePlotSchema.safeParse(parsedValue);

    if (!parsedRoute.success) {
      console.error("Invalid route plot state found in localStorage; resetting.", parsedRoute.error, parsedValue);
      clearStoredRoutePlot();
      return null;
    }

    return parsedRoute.data;
  } catch (error) {
    console.error("Failed to parse route plot state from localStorage; resetting.", error);
    clearStoredRoutePlot();
    return null;
  }
}

export function saveStoredRoutePlot(routePlot: StoredRoutePlot): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(routePlot));
}

export function clearStoredRoutePlot(): void {
  localStorage.removeItem(STORAGE_KEY);
}
