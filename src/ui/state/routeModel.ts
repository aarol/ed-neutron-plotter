import { createModel, Signal, signal } from "@preact/signals"
import { createContext, type Context } from "preact"
import type { StarSystem } from "../types"
import { loadStoredRoutePlot, saveStoredRoutePlot } from "./localStorage";

export interface RouteState {
  nodes: Signal<StarSystem[]>;
  progress: Signal<number>;
  setRoute: (newNodes: StarSystem[]) => void;
  clearRoute: () => void;
  setProgress: (nextProgress: number) => void;
}

export const RouteModel = createModel<RouteState>(() => {

  const storedRoute = loadStoredRoutePlot();

  const nodes = signal<StarSystem[]>(storedRoute?.nodes ?? [])
  const progress = signal(storedRoute?.progress ?? 0)

  return {
    nodes,
    progress,

    setRoute(newNodes: StarSystem[]) {
      nodes.value = newNodes
      progress.value = 0
      saveStoredRoutePlot({nodes: newNodes, progress: progress.value})
    },
    clearRoute() {
      nodes.value = []
      progress.value = 0
      saveStoredRoutePlot({nodes: nodes.value, progress: progress.value})
    },
    setProgress(nextProgress: number) {
      progress.value = nextProgress
      saveStoredRoutePlot({nodes: nodes.value, progress: progress.value})
    }
  }
})

export const RouteContext: Context<RouteState | null> = createContext<RouteState | null>(null)
