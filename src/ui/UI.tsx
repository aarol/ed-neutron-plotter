import { useContext, useState } from "preact/hooks";
import { JournalDialog } from "./JournalDialog";
import { RouteListPanel } from "./RouteListPanel";
import { RouteDialog } from "./RouteDialog";
import { SearchBar } from "./SearchBar";
import type { SearchSubmitOptions } from "./SearchBox";
import { TargetInfo } from "./TargetInfo";
import { useToast } from "./toast";
import type { RouteConfig, StarSystem } from "./types";
import { RouteContext } from "./state/routeModel";
import { Show } from "@preact/signals/utils";
import { signal } from "@preact/signals";
import { JournalContext } from "./state/journalModel";
import { loadStoredFocusedSystem } from "./state/localStorage";

export interface UIProps {
  onGenerateRoute: (config: RouteConfig) => Promise<StarSystem[]>;
  onSelectTarget: (query: string) => Promise<StarSystem | null>;
  autocomplete: (word: string) => string[];
}

export const showJournalDialog = signal(false);
export const showRouteDialog = signal(false);

export const focusedSystem = signal<StarSystem>(loadStoredFocusedSystem() ?? { name: "Sol", coords: { x: 0, y: 0, z: 0 } });

export function UI({ onGenerateRoute, onSelectTarget, autocomplete }: UIProps) {
  const { showError } = useToast();
  const journalState = useContext(JournalContext)!;
  const routeState = useContext(RouteContext)!;
  const [routeDialogToValue, setRouteDialogToValue] = useState("");

  const openRouteDialog = (word: string) => {
    setRouteDialogToValue(word);
    showRouteDialog.value = true;
  };

  const handleSearch = async (query: string, options: SearchSubmitOptions) => {
    try {
      const nextTarget = await onSelectTarget(query);
      if (!nextTarget) {
        showError(`Star not found: ${query}`);
        return;
      }

      focusedSystem.value = nextTarget;
      if (options.openRoute) {
        openRouteDialog(query);
      }
    } catch (error) {
      showError(error instanceof Error ? error.message : "Failed to find star.");
    }
  };

  const handleGenerateRoute = async (config: RouteConfig) => {
    try {
      const nextRouteNodes = await onGenerateRoute(config);
      routeState.setRoute(nextRouteNodes);
    } catch (error) {
      showError(error instanceof Error ? error.message : "Failed to generate route.");
    }
  };

  return (
    <>
      <SearchBar
        onClickRoute={openRouteDialog}
        onOpenJournal={async () => {
          if (journalState.enabled.value) {
            journalState.stop();
            return;
          }
          showJournalDialog.value = true;
        }}
        onSearch={(query, options) => {
          void handleSearch(query, options);
        }}
        onSuggest={autocomplete}
      />

      <TargetInfo isRoutePanelOpen={showRouteDialog.value} onOpenRoute={openRouteDialog} />

      <Show when={() => routeState.nodes.value.length > 0}>
        <RouteListPanel />
      </Show>

      <RouteDialog
        initialToValue={routeDialogToValue}
        onSubmit={handleGenerateRoute}
        onSuggest={autocomplete}
      />

      <JournalDialog />
    </>
  );
};