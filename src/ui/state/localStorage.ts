import { z } from "zod";

const starSystemSchema = z.object({
  name: z.string(),
  coords: z.object({
    x: z.number(),
    y: z.number(),
    z: z.number(),
  }),
});

const routeNodeSchema = z.object({
  system: starSystemSchema,
  distance: z.number(),
  refuel: z.boolean(),
  isNeutron: z.boolean(),
});


const storedRoutePlotSchema = z.object({
  nodes: z.array(routeNodeSchema),
  progress: z.number().int().min(0),
});

export type StoredRoutePlot = z.infer<typeof storedRoutePlotSchema>;

const ROUTE_STORAGE_KEY = "ed-galaxy.route-plot";

export function loadStoredRoutePlot() {
  return loadFromLocalStorage<StoredRoutePlot>(ROUTE_STORAGE_KEY, storedRoutePlotSchema);
}

export function saveStoredRoutePlot(routePlot: StoredRoutePlot) {
  localStorage.setItem(ROUTE_STORAGE_KEY, JSON.stringify(routePlot));
}

export function clearStoredRoutePlot(): void {
  localStorage.removeItem(ROUTE_STORAGE_KEY);
}

function loadFromLocalStorage<T>(key: string, schema: z.ZodType<T>): T | null {
  const rawValue = localStorage.getItem(key);
  if (!rawValue) {
    return null;
  }

  try {
    const parsedValue = JSON.parse(rawValue);
    const parsedData = schema.safeParse(parsedValue);
    if (!parsedData.success) {
      console.error(`Invalid data found in localStorage for key "${key}"; resetting.`, parsedData.error, parsedValue);
      localStorage.removeItem(key);
      return null;
    }
    return parsedData.data;
  } catch (error) {
    console.error(`Failed to parse data from localStorage for key "${key}"; resetting.`, error);
    localStorage.removeItem(key);
    return null;
  }
}

export type StoredFocusedSystem = z.infer<typeof starSystemSchema>;

const FOCUSED_SYSTEM_STORAGE_KEY = "ed-galaxy.last-focused-system";

export function loadStoredFocusedSystem(): StoredFocusedSystem | null {
  return loadFromLocalStorage<StoredFocusedSystem>(FOCUSED_SYSTEM_STORAGE_KEY, starSystemSchema);
}

export function saveStoredFocusedSystem(system: StoredFocusedSystem): void {
  localStorage.setItem(FOCUSED_SYSTEM_STORAGE_KEY, JSON.stringify(system));
}

export function clearStoredFocusedSystem(): void {
  localStorage.removeItem(FOCUSED_SYSTEM_STORAGE_KEY);
}
