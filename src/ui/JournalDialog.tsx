import { useEffect, useRef, useState } from "preact/hooks";
import styles from "./JournalDialog.module.css";

interface JournalDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onInitialize: () => Promise<void>;
}

export function JournalDialog({ isOpen, onClose, onInitialize }: JournalDialogProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const [isInitializing, setIsInitializing] = useState(false);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) {
      return;
    }

    if (isOpen && !dialog.open) {
      dialog.showModal();
      return;
    }

    if (!isOpen && dialog.open) {
      dialog.close();
    }
  }, [isOpen]);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) {
      return;
    }

    const handleClose = () => {
      onClose();
    };

    dialog.addEventListener("close", handleClose);
    return () => {
      dialog.removeEventListener("close", handleClose);
    };
  }, [onClose]);

  const handleInitialize = async () => {
    setIsInitializing(true);
    try {
      await onInitialize();
      dialogRef.current?.close();
    } catch (error) {
      console.error("Error accessing journal directory:", error);
    } finally {
      setIsInitializing(false);
    }
  };

  return (
    <dialog className={styles.dialog} ref={dialogRef}>
      <div className={styles.header}>
        <h3 className={styles.title}>Elite Dangerous Journal</h3>
        <button className={styles.closeBtn} onClick={() => dialogRef.current?.close()} title="Close" type="button">
          ✕
        </button>
      </div>

      <div className={styles.body}>
        <p className={styles.instruction}>
          Navigate to <span className={styles.path}>C:\Users\&lt;Username&gt;\Saved Games\Frontier Developments\Elite Dangerous</span> and select the directory.
        </p>

        <button className={styles.selectBtn} disabled={isInitializing} onClick={() => void handleInitialize()} type="button">
          Select Directory
        </button>
      </div>
    </dialog>
  );
}