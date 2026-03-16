import { useEffect, useRef, useState } from "preact/hooks";
import { SearchBox, type SearchBoxHandle } from "./SearchBox";
import type { RouteConfig } from "./types";
import styles from "./RouteDialog.module.css";

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
    <div className={styles.panel}>
      <div className={styles.header}>
        <h2 className={styles.title}>Configure Route</h2>
        <button className={styles.closeBtn} onClick={onClose} type="button">
          ×
        </button>
      </div>

      <div className={styles.formSection}>
        <label className={styles.label}>From:</label>
        <SearchBox
          autoFocus
          className={styles.dialogSearchBox}
          inputClassName={styles.dialogSearchInput}
          onSuggest={onSuggest}
          onValueChange={setFrom}
          placeholder="Enter starting star..."
          ref={fromRef}
          showRouteButton={false}
          value={from}
        />
      </div>

      <div className={styles.formSection}>
        <label className={styles.label}>To:</label>
        <SearchBox
          className={styles.dialogSearchBox}
          inputClassName={styles.dialogSearchInput}
          onSuggest={onSuggest}
          onValueChange={setTo}
          placeholder="Enter destination star..."
          showRouteButton={false}
          value={to}
        />
      </div>

      <div className={styles.checkboxSection}>
        <label className={styles.checkboxLabel}>
          <input
            checked={alreadySupercharged}
            className={styles.checkbox}
            onInput={(event) => setAlreadySupercharged((event.currentTarget as HTMLInputElement).checked)}
            type="checkbox"
          />
          Already supercharged
        </label>
      </div>

      <div className={styles.buttonGroup}>
        <button className={styles.cancelBtn} onClick={onClose} type="button">
          Cancel
        </button>
        <button className={styles.generateBtn} disabled={isSubmitting} onClick={() => void handleGenerate()} type="button">
          Generate Route
        </button>
      </div>
    </div>
  );
}