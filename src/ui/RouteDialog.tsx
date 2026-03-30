import { useEffect, useRef, useState } from "preact/hooks";
import { SearchBox, type SearchBoxHandle } from "./SearchBox";
import { Button } from "./components/Button";
import type { RouteConfig } from "./types";
import { uiTheme } from "./theme";
import { effect } from "@preact/signals";
import { showRouteDialog } from "./UI";

interface RouteDialogProps {
  initialToValue?: string;
  onSubmit: (config: RouteConfig) => Promise<void> | void;
  onSuggest: (word: string) => string[];
}

export function RouteDialog({
  initialToValue = "",
  onSubmit,
  onSuggest,
}: RouteDialogProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const fromRef = useRef<SearchBoxHandle>(null);
  const [from, setFrom] = useState("");
  const [to, setTo] = useState(initialToValue);
  const [alreadySupercharged, setAlreadySupercharged] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const closeDialog = () => {
    showRouteDialog.value = false;
  };

  useEffect(() => {
    setTo(initialToValue);
  }, [initialToValue]);

  useEffect(() => {
    const dispose = effect(() => {
      const dialog = dialogRef.current;
      if (showRouteDialog.value) {
        setFrom("");
        setAlreadySupercharged(false);
        setIsSubmitting(false);
        if (!dialog?.open) {
          dialog?.showModal();
        }
      } else {
        dialog?.close();
      }
    });

    return () => {
      dispose();
    };
  }, []);

  const canGenerate = from.trim().length > 0 && to.trim().length > 0;

  const handleGenerate = async () => {
    const config: RouteConfig = {
      from: from.trim(),
      to: to.trim(),
      alreadySupercharged,
    };

    if (!config.from || !config.to) {
      return;
    }

    setIsSubmitting(true);
    try {
      await onSubmit(config);
      closeDialog();
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <dialog
      className={`${uiTheme.glassPanel} fixed left-1/2 top-1/2 z-999 m-0 w-[min(92vw,420px)] -translate-x-1/2 -translate-y-1/2 p-4`}
      onClose={closeDialog}
      ref={dialogRef}
    >
      <div className={`${uiTheme.panelHeader} mb-3`}>
        <h2 className={uiTheme.panelTitle}>Plot route</h2>
        <Button onClick={closeDialog} variant="icon">
          ×
        </Button>
      </div>

      <div className="relative mb-2.5">
        <label className="mb-1 block text-[11px] font-medium text-white/70">From:</label>
        <SearchBox
          autoFocus
          className="w-full"
          inputClassName={uiTheme.compactInput}
          onSuggest={onSuggest}
          onValueChange={setFrom}
          placeholder="Enter starting star..."
          ref={fromRef}
          showRouteButton={false}
          value={from}
        />
      </div>

      <div className="relative mb-2.5">
        <label className="mb-1 block text-[11px] font-medium text-white/70">To:</label>
        <SearchBox
          className="w-full"
          inputClassName={uiTheme.compactInput}
          onSuggest={onSuggest}
          onValueChange={setTo}
          placeholder="Enter destination star..."
          showRouteButton={false}
          value={to}
        />
      </div>

      <div className="mb-3">
        <label className="flex cursor-pointer items-center text-xs text-white/90">
          <input
            checked={alreadySupercharged}
            className="mr-2 h-3.5 w-3.5 cursor-pointer border border-white/35 bg-black/40 accent-space-accent-strong"
            onInput={(event) => setAlreadySupercharged((event.currentTarget as HTMLInputElement).checked)}
            type="checkbox"
          />
          Already supercharged
        </label>
      </div>

      <div className="flex justify-end gap-2">
        <Button onClick={closeDialog} variant="secondary">
          Cancel
        </Button>
        <Button
          className={!canGenerate ? "border-white/25 bg-white/10 text-white/45 shadow-none hover:border-white/25 hover:bg-white/10" : undefined}
          disabled={isSubmitting || !canGenerate}
          onClick={() => void handleGenerate()}
          variant="primary"
        >
          Generate Route
        </Button>
      </div>
    </dialog>
  );
}