import { createModel, effect, Signal, signal } from "@preact/signals"
import { createContext, type Context } from "preact"
import type { StarSystem } from "../types"
import { loadStoredRoutePlot, saveStoredRoutePlot } from "./localStorage";

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
  markVisitedSystem: (systemName: string) => void;
}

export const RouteModel = createModel<RouteState>(() => {

  const storedRoute = loadStoredRoutePlot();

  const nodes = signal<RouteNode[]>(storedRoute?.nodes ?? [])
  const progress = signal(storedRoute?.progress ?? 0)

  effect(() => {
    saveStoredRoutePlot({ nodes: nodes.value, progress: progress.value })
  })

  return {
    nodes,
    progress,

    setRoute(newNodes: RouteNode[]) {
      nodes.value = newNodes
      progress.value = 0
    },
    clearRoute() {
      nodes.value = []
      progress.value = 0
    },
    setProgress(nextProgress: number) {
      progress.value = nextProgress
    },
    markVisitedSystem(systemName: string) {
      if (!systemName || nodes.value.length === 0) {
        return;
      }

      const routeIndex = nodes.value.findIndex((node) => node.system.name === systemName);

      if (routeIndex === -1) {
        return;
      }

      const nextProgress = routeIndex + 1;
      if (nextProgress > progress.value) {
        progress.value = nextProgress;
      }
    }
  }
})

export const RouteContext: Context<RouteState | null> = createContext<RouteState | null>(null)
