import { useContext, useRef, useState } from "preact/hooks";
import { Button } from "./components/Button";
import { uiTheme } from "./theme";
import { useToast } from "./toast";
import { JournalContext } from "./state/journalModel";
import { effect } from "@preact/signals";
import { showJournalDialog } from "./UI";

interface JournalDialogProps {
}

export function JournalDialog({ }: JournalDialogProps) {
  const { showError } = useToast();
  const dialogRef = useRef<HTMLDialogElement>(null);
  const [isInitializing, setIsInitializing] = useState(false);
  const journalPickerPath = "%userprofile%\\Saved Games\\Frontier Developments\\Elite Dangerous";

  const supportsObserverApi = "FileSystemObserver" in window // If the observer API is supported, then the file system access API must also be supported

  const journalState = useContext(JournalContext)!;

  effect(() => {
    const dialog = dialogRef.current;
    if (showJournalDialog.value) {
      if (!dialog?.open) dialog?.showModal();
    } else {
      dialog?.close();
    }
  })

  const closeDialog = () => showJournalDialog.value = false;

  const handleInitialize = async () => {
    setIsInitializing(true);
    try {
      await journalState.init();
      closeDialog();
    } catch (error) {
      console.error("Error accessing journal directory:", error);
      showError(error instanceof Error ? error.message : "Failed to initialize journal tracking.");
    } finally {
      setIsInitializing(false);
    }
  };

  return (
    <dialog
      className={`${uiTheme.glassPanel} fixed left-1/2 top-1/2 m-0 w-[min(92vw,680px)] -translate-x-1/2 -translate-y-1/2 p-0`}
      ref={dialogRef}
      onClose={closeDialog}
    >
      <div className={`${uiTheme.panelHeader} border-b border-white/10 px-4.5 pb-3 pt-4`}>
        <h3 className={uiTheme.panelTitle}>Track in-game location</h3>
        <Button onClick={() => dialogRef.current?.close()} title="Close" variant="icon">
          ✕
        </Button>
      </div>

      {supportsObserverApi ? <>
        <div className="flex flex-col gap-4 px-4.5 pb-5 pt-4.5">
          <p className="m-0 text-[13px] leading-relaxed text-white/80">
            Elite Dangerous has a "journal" feature that logs your in-game location into a file located at <span className="inline-block border border-white/20 bg-white/10 px-1.5 py-0.5 font-mono text-[11.5px] whitespace-nowrap text-[#b4c8ff]">C:\Users\&lt;Username&gt;\Saved Games\Frontier Developments\Elite Dangerous</span>.
          </p>

          <p className="m-0 text-[13px] leading-relaxed text-white/80">
            To enable real-time tracking of your in-game location, please click the "Select Directory" button below, navigate to the directory shown above and click "Select".

            You can also copy and paste the following path into the file picker dialog to navigate there directly:{" "}
            <span className="inline-flex items-center gap-1 align-middle">
              <span className="inline-block border border-white/20 bg-white/10 px-1.5 py-0.5 font-mono text-[11.5px] whitespace-nowrap text-[#b4c8ff]">
                {journalPickerPath}
              </span>
              <Button
                aria-label="Copy journal path"
                className="inline-flex h-5 w-5 items-center justify-center border border-white/30 bg-white/8 text-white/80 hover:bg-white/15 hover:text-white"
                onClick={() => void navigator.clipboard.writeText(journalPickerPath)}
                title="Copy path"
                variant="plain"
              >
                <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" viewBox="0 0 24 24">
                  <rect height="13" rx="2" ry="2" width="13" x="9" y="9" />
                  <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                </svg>
              </Button>
            </span>
            .
          </p>

          <Button className="self-start px-4 py-2 text-[13px]" disabled={isInitializing} onClick={() => void handleInitialize()} variant="primary">
            Select Directory
          </Button>
        </div>
      </> : <>
        <div className="flex flex-col gap-4 px-4.5 pb-5 pt-4.5">
          <p className="m-0 text-[13px] leading-relaxed text-white/80">
            Your browser does not support the File System Access API, which is required for real-time tracking of your in-game location. Please use a compatible browser (like Chrome or Edge) to access this feature.
          </p>
        </div>
      </>}


    </dialog>
  );
}