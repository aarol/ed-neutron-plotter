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
import { signal, useSignal } from "@preact/signals";
import { JournalContext } from "./state/journalModel";
import { loadStoredFocusedSystem } from "./state/localStorage";
import { HamburgerMenu } from "./HamburgerMenu";
import { ImportSpanshDialog } from "./ImportSpanshDialog";

export interface UIProps {
  onGenerateRoute: (config: RouteConfig) => Promise<StarSystem[]>;
  onSelectTarget: (query: string) => Promise<StarSystem | null>;
  autocomplete: (word: string) => string[];
}

export const focusedSystem = signal<StarSystem>(loadStoredFocusedSystem() ?? { name: "Sol", coords: { x: 0, y: 0, z: 0 } });

export function UI({ onGenerateRoute, onSelectTarget, autocomplete }: UIProps) {
  const { showError } = useToast();
  const journalState = useContext(JournalContext)!;
  const routeState = useContext(RouteContext)!;
  const [routeDialogToValue, setRouteDialogToValue] = useState("");
  const [routeDialogFromValue, setRouteDialogFromValue] = useState("");
  const openDialog = useSignal<"route" | "journal" | "importSpansh" | null>(null);

  const openRouteDialog = (word: string) => {
    setRouteDialogFromValue(word);
    setRouteDialogToValue("");
    openDialog.value = "route";
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
    openDialog.value = null;
    try {
      const nextRouteNodes = await onGenerateRoute(config);
      routeState.setRoute(nextRouteNodes.map(system => ({
        system,
        distance: 0, // Placeholder, you can calculate actual distances if needed
        refuel: true, // Placeholder, you can determine refuel points if needed
        isNeutron: true, // Placeholder, you can determine neutron points if needed
      })));
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
          openDialog.value = "journal";
        }}
        onSearch={(query, options) => {
          void handleSearch(query, options);
        }}
        onSuggest={autocomplete}
      />

      <HamburgerMenu onImportRouteFromSpansh={() => {
        openDialog.value = "importSpansh";
      }} />

      <TargetInfo isRoutePanelOpen={openDialog.value === "route"} onOpenRoute={openRouteDialog} />

      <Show when={() => routeState.nodes.value.length > 0}>
        <RouteListPanel />
      </Show>

      <RouteDialog
        dialogOpen={openDialog.value === "route"}
        initialFromValue={routeDialogFromValue}
        initialToValue={routeDialogToValue}
        onSubmit={handleGenerateRoute}
        onSuggest={autocomplete}
        onClose={() => openDialog.value = null}
      />

      <ImportSpanshDialog
        dialogOpen={openDialog.value === "importSpansh"}
        onClose={() => openDialog.value = null}
      />

      <JournalDialog
        dialogOpen={openDialog.value === "journal"}
        onClose={() => openDialog.value = null}
      />
    </>
  );
};