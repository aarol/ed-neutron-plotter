import { forwardRef } from "preact/compat";
import { useContext, useImperativeHandle, useState } from "preact/hooks";
import { JournalDialog } from "./JournalDialog";
import { RouteListPanel } from "./RouteListPanel";
import { RouteDialog } from "./RouteDialog";
import { SearchBar } from "./SearchBar";
import type { SearchSubmitOptions } from "./SearchBox";
import { TargetInfo } from "./TargetInfo";
import { useToast } from "./toast";
import type { RouteConfig, RouteNode, TargetInfoState } from "./types";
import { RouteContext } from "./state/routeModel";
import { Show } from "@preact/signals/utils";
import { computed } from "@preact/signals";

export interface UIProps {
  onGenerateRoute: (config: RouteConfig) => Promise<RouteNode[]>;
  onRouteSelectionChange: (index: number) => void;
  onRestoreStoredRoute: (nodes: RouteNode[], progress: number) => void;
  onInitializeJournal: () => Promise<void>;
  onStopJournalTracking: () => void;
  onSelectTarget: (query: string) => Promise<TargetInfoState | null>;
  onSuggest: (word: string) => string[];
}

export interface UIHandle {
  openRouteDialog: (word: string) => void;
  setTargetInfo: (target: TargetInfoState) => void;
}

export const UI = forwardRef<UIHandle, UIProps>(function UI(
  { onGenerateRoute, onInitializeJournal, onSelectTarget, onStopJournalTracking, onSuggest },
  ref,
) {
  const { showError } = useToast();
  const [target, setTarget] = useState<TargetInfoState>({ name: "Sol", x: 0, y: 0, z: 0 });
  const [isJournalOpen, setIsJournalOpen] = useState(false);
  const [isJournalTracking, setIsJournalTracking] = useState(false);
  const routeState = useContext(RouteContext)!;
  const showRouteListPanel = computed(() => routeState.nodes.value.length > 0);

  const [routeDialogState, setRouteDialogState] = useState({
    isOpen: false,
    fromValue: "",
    toValue: "",
    alreadySupercharged: false,
  });

  const openRouteDialog = (word: string) => {
    setRouteDialogState((currentState) => ({
      ...currentState,
      isOpen: true,
      toValue: word,
    }));
  };

  useImperativeHandle(
    ref,
    () => ({
      openRouteDialog,
      setTargetInfo: (nextTarget: TargetInfoState) => {
        setTarget(nextTarget);
      },
    }),
    [],
  );

  const handleSearch = async (query: string, options: SearchSubmitOptions) => {
    try {
      const nextTarget = await onSelectTarget(query);
      if (!nextTarget) {
        showError(`Star not found: ${query}`);
        return;
      }

      setTarget(nextTarget);
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

    setRouteDialogState({
      isOpen: false,
      fromValue: config.from,
      toValue: config.to,
      alreadySupercharged: config.alreadySupercharged,
    });
  };

  return (
    <>
      <SearchBar
        onClickRoute={openRouteDialog}
        isJournalTracking={isJournalTracking}
        onOpenJournal={() => {
          if (isJournalTracking) {
            onStopJournalTracking();
            setIsJournalTracking(false);
            setIsJournalOpen(false);
            return;
          }

          setIsJournalOpen(true);
        }}
        onSearch={(query, options) => {
          void handleSearch(query, options);
        }}
        onSuggest={onSuggest}
      />

      <TargetInfo isRoutePanelOpen={routeDialogState.isOpen} onOpenRoute={openRouteDialog} target={target} />

      <Show when={showRouteListPanel}>
        <RouteListPanel />
      </Show>

      <RouteDialog
        initialFromValue={routeDialogState.fromValue}
        initialSupercharged={routeDialogState.alreadySupercharged}
        initialToValue={routeDialogState.toValue}
        isOpen={routeDialogState.isOpen}
        onClose={() => setRouteDialogState((currentState) => ({ ...currentState, isOpen: false }))}
        onSubmit={handleGenerateRoute}
        onSuggest={onSuggest}
      />

      <JournalDialog
        isOpen={isJournalOpen}
        onClose={() => setIsJournalOpen(false)}
        onInitialize={async () => {
          await onInitializeJournal();
          setIsJournalTracking(true);
        }}
      />
    </>
  );
});