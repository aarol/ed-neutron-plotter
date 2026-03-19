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
  const journalPickerPath = "%userprofile%\\Saved Games\\Frontier Developments\\Elite Dangerous";

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
    <dialog
      className={`${uiTheme.glassPanel} fixed left-1/2 top-1/2 m-0 w-[min(92vw,680px)] -translate-x-1/2 -translate-y-1/2 p-0`}
      ref={dialogRef}
    >
      <div className={`${uiTheme.panelHeader} border-b border-white/10 px-4.5 pb-3 pt-4`}>
        <h3 className={uiTheme.panelTitle}>Track in-game location</h3>
        <button className={uiTheme.iconButton} onClick={() => dialogRef.current?.close()} title="Close" type="button">
          ✕
        </button>
      </div>

      <div className="flex flex-col gap-4 px-4.5 pb-5 pt-4.5">
        <p className="m-0 text-[13px] leading-relaxed text-white/80">
          Elite Dangerous has a "journal" feature that logs your in-game location into a file located at <span className="inline-block border border-white/20 bg-white/10 px-1.5 py-0.5 font-mono text-[11.5px] whitespace-nowrap text-[#b4c8ff]">C:\Users\&lt;Username&gt;\Saved Games\Frontier Developments\Elite Dangerous</span>.
        </p>

        <p className="m-0 text-[13px] leading-relaxed text-white/80">
          Via the <a href="https://developer.mozilla.org/en-US/docs/Web/API/FileSystemObserver">FileSystemObserver</a> API, browsers can listen for changes in files such as the ED journal.
        </p>

        <p className="m-0 text-[13px] leading-relaxed text-white/80">
          To enable real-time tracking of your in-game location, please click the "Select Directory" button below and navigate to the directory above and click "Select".

          You can also copy and paste the following path into the file picker dialog to navigate there directly:{" "}
          <span className="inline-flex items-center gap-1 align-middle">
            <span className="inline-block border border-white/20 bg-white/10 px-1.5 py-0.5 font-mono text-[11.5px] whitespace-nowrap text-[#b4c8ff]">
              {journalPickerPath}
            </span>
            <button
              aria-label="Copy journal path"
              className="inline-flex h-5 w-5 items-center justify-center border border-white/30 bg-white/8 text-white/80 transition hover:bg-white/15 hover:text-white"
              onClick={() => void navigator.clipboard.writeText(journalPickerPath)}
              title="Copy path"
              type="button"
            >
              <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" viewBox="0 0 24 24">
                <rect height="13" rx="2" ry="2" width="13" x="9" y="9" />
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
              </svg>
            </button>
          </span>
          .
        </p>

        <button className={`${uiTheme.primaryButton} self-start px-4 py-2 text-[13px]`} disabled={isInitializing} onClick={() => void handleInitialize()} type="button">
          Select Directory
        </button>
      </div>
    </dialog>
  );
}