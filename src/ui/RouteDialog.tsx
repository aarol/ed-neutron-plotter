import { useEffect, useRef, useState } from "preact/hooks";
import { SearchBox, type SearchBoxHandle } from "./SearchBox";
import { Button } from "./components/Button";
import type { RouteConfig } from "./types";
import { uiTheme } from "./theme";

interface RouteDialogProps {
  isOpen: boolean;
  initialFromValue?: string;
  initialToValue?: string;
  initialSupercharged?: boolean;
  onClose: () => void;
  onSubmit: (config: RouteConfig) => Promise<void> | void;
  onSuggest: (word: string) => string[];
}

export function RouteDialog({
  initialFromValue = "",
  initialSupercharged = false,
  initialToValue = "",
  isOpen,
  onClose,
  onSubmit,
  onSuggest,
}: RouteDialogProps) {
  const fromRef = useRef<SearchBoxHandle>(null);
  const [from, setFrom] = useState(initialFromValue);
  const [to, setTo] = useState(initialToValue);
  const [alreadySupercharged, setAlreadySupercharged] = useState(initialSupercharged);
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    if (!isOpen) {
      return;
    }

    setFrom(initialFromValue);
    setTo(initialToValue);
    setAlreadySupercharged(initialSupercharged);
    window.setTimeout(() => {
      fromRef.current?.focus();
    }, 0);
  }, [initialFromValue, initialSupercharged, initialToValue, isOpen]);

  useEffect(() => {
    if (!isOpen) {
      return;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [isOpen, onClose]);

  if (!isOpen) {
    return null;
  }

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
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className={`${uiTheme.glassPanel} fixed left-5 top-20 z-999 w-280px p-4`}>
      <div className={`${uiTheme.panelHeader} mb-3`}>
        <h2 className={uiTheme.panelTitle}>Configure Route</h2>
        <Button onClick={onClose} variant="icon">
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
        <Button onClick={onClose} variant="secondary">
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
    </div>
  );
}