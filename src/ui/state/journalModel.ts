import { createModel, Signal, signal } from "@preact/signals";
import { createContext, type Context } from "preact";
import { Journal } from "../../journal/journal";
import type { StarSystem } from "../types";

export interface JournalState {
  lastSystem: Signal<StarSystem | null>;
  init: () => Promise<void>;
  enabled: Signal<boolean>;
  stop: () => void;
}

export const JournalModel = createModel<JournalState>(() => {

  const lastSystem = signal<StarSystem | null>(null)

  const journal = new Journal({
    onNewLocation: system => {
      lastSystem.value = system;
    }
  })
  const enabled = signal(false);

  return {
    lastSystem,
    async init() {
      await journal.init();
      enabled.value = true;
    },
    stop() {
      journal.stopTracking();
      enabled.value = false;
    },
    enabled,
  }
})

export const JournalContext: Context<JournalState | null> = createContext<JournalState | null>(null)
