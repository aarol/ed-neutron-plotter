import { z } from "zod";

import type { TargetInfoState } from "./types";

const focusedSystemSchema = z.object({
  name: z.string(),
  x: z.number(),
  y: z.number(),
  z: z.number(),
});

export type StoredFocusedSystem = z.infer<typeof focusedSystemSchema>;

const STORAGE_KEY = "ed-galaxy.last-focused-system";

export function loadStoredFocusedSystem(): StoredFocusedSystem | null {
  const rawValue = localStorage.getItem(STORAGE_KEY);
  if (!rawValue) {
    return null;
  }

  try {
    const parsedValue = JSON.parse(rawValue);
    const parsedSystem = focusedSystemSchema.safeParse(parsedValue);

    if (!parsedSystem.success) {
      console.error("Invalid focused system found in localStorage; resetting.", parsedSystem.error, parsedValue);
      clearStoredFocusedSystem();
      return null;
    }

    return parsedSystem.data;
  } catch (error) {
    console.error("Failed to parse focused system from localStorage; resetting.", error);
    clearStoredFocusedSystem();
    return null;
  }
}

export function saveStoredFocusedSystem(system: TargetInfoState): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(system));
}

export function clearStoredFocusedSystem(): void {
  localStorage.removeItem(STORAGE_KEY);
}
