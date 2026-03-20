import { forwardRef } from "preact/compat";
import { useImperativeHandle, useState } from "preact/hooks";
import { JournalDialog } from "./JournalDialog";
import { RouteListPanel } from "./RouteListPanel";
import { RouteDialog } from "./RouteDialog";
import { SearchBar } from "./SearchBar";
import type { SearchSubmitOptions } from "./SearchBox";
import { TargetInfo } from "./TargetInfo";
import type { RouteConfig, RouteNode, TargetInfoState } from "./types";

export interface UIProps {
  onGenerateRoute: (config: RouteConfig) => Promise<RouteNode[]>;
  onRouteSelectionChange: (checkedByIndex: boolean[]) => void;
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
  { onGenerateRoute, onInitializeJournal, onRouteSelectionChange, onSelectTarget, onStopJournalTracking, onSuggest },
  ref,
) {
  const [target, setTarget] = useState<TargetInfoState>({ name: "Sol", x: 0, y: 0, z: 0 });
  const [isJournalOpen, setIsJournalOpen] = useState(false);
  const [isJournalTracking, setIsJournalTracking] = useState(false);
  const [isRouteListPanelVisible, setIsRouteListPanelVisible] = useState(false);
  const [routeNodes, setRouteNodes] = useState<RouteNode[]>([]);
  const [routeCheckedByIndex, setRouteCheckedByIndex] = useState<boolean[]>([]);
  const [routeDialogState, setRouteDialogState] = useState({
    isOpen: false,
    fromValue: "",
    toValue: "",
    alreadySupercharged: false,
  });

  const openRouteDialog = (word: string) => {
    setIsRouteListPanelVisible(false);
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
    const nextTarget = await onSelectTarget(query);
    if (!nextTarget) {
      window.alert(`Star not found: ${query}`);
      return;
    }

    setTarget(nextTarget);
    if (options.openRoute) {
      openRouteDialog(query);
    }
  };

  const handleGenerateRoute = async (config: RouteConfig) => {
    const nextRouteNodes = await onGenerateRoute(config);

    if (nextRouteNodes.length > 0) {
      setRouteNodes(nextRouteNodes);
      const nextCheckedByIndex = nextRouteNodes.map(() => false);
      setRouteCheckedByIndex(nextCheckedByIndex);
      onRouteSelectionChange(nextCheckedByIndex);
      setIsRouteListPanelVisible(true);
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

      <RouteListPanel
        checkedByIndex={routeCheckedByIndex}
        nodes={routeNodes}
        onCheckedByIndexChange={(nextCheckedByIndex) => {
          setRouteCheckedByIndex(nextCheckedByIndex);
          onRouteSelectionChange(nextCheckedByIndex);
        }}
        visible={isRouteListPanelVisible}
      />

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