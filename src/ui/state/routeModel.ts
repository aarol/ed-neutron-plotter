import { createModel, Signal, signal } from "@preact/signals"
import { createContext, type Context } from "preact"
import type { StarSystem } from "../types"
import { clearStoredRoutePlot, loadStoredRoutePlot, saveStoredRoutePlot } from "./localStorage";

export type RouteNode = {
  system: StarSystem;
  distance: number;
  refuel: boolean;
  isNeutron: boolean;
}

export interface RouteState {
  nodes: Signal<RouteNode[]>;
  progress: Signal<number>;
  setRoute: (newNodes: RouteNode[]) => void;
  clearRoute: () => void;
  setProgress: (nextProgress: number) => void;
}

export const RouteModel = createModel<RouteState>(() => {

  const storedRoute = loadStoredRoutePlot();

  const nodes = signal<RouteNode[]>(storedRoute?.nodes ?? [])
  const progress = signal(storedRoute?.progress ?? 0)

  return {
    nodes,
    progress,

    setRoute(newNodes: RouteNode[]) {
      nodes.value = newNodes
      progress.value = 0
      saveStoredRoutePlot({nodes: newNodes, progress: progress.value})
    },
    clearRoute() {
      nodes.value = []
      progress.value = 0
      clearStoredRoutePlot()
    },
    setProgress(nextProgress: number) {
      progress.value = nextProgress
      saveStoredRoutePlot({nodes: nodes.value, progress: progress.value})
    }
  }
})

export const RouteContext: Context<RouteState | null> = createContext<RouteState | null>(null)
