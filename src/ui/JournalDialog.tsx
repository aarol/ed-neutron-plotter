import { useEffect, useRef, useState } from "preact/hooks";
import { uiTheme } from "./theme";

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
    <dialog className={`${uiTheme.glassPanel} w-90 p-0`} ref={dialogRef}>
      <div className={`${uiTheme.panelHeader} border-b border-white/10 px-4.5 pb-3 pt-4`}>
        <h3 className={uiTheme.panelTitle}>Elite Dangerous Journal</h3>
        <button className={uiTheme.iconButton} onClick={() => dialogRef.current?.close()} title="Close" type="button">
          ✕
        </button>
      </div>

      <div className="flex flex-col gap-4 px-4.5 pb-5 pt-4.5">
        <p className="m-0 text-[13px] leading-relaxed text-white/80">
          Navigate to <span className="inline-block border border-white/20 bg-white/10 px-1.5 py-0.5 font-mono text-[11.5px] whitespace-nowrap text-[#b4c8ff]">C:\Users\&lt;Username&gt;\Saved Games\Frontier Developments\Elite Dangerous</span> and select the directory.
        </p>

        <button className={`${uiTheme.primaryButton} self-start px-4 py-2 text-[13px]`} disabled={isInitializing} onClick={() => void handleInitialize()} type="button">
          Select Directory
        </button>
      </div>
    </dialog>
  );
}