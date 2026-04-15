import { useContext, useEffect, useRef, useState } from "preact/hooks";
import { Button } from "./components/Button";
import { uiTheme } from "./theme";
import { parseSpanshHtml } from "../spansh-import";
import { useToast } from "./toast";
import { RouteContext } from "./state/routeModel";

type ImportSpanshDialogProps = {
  dialogOpen: boolean;
  onClose: () => void;
}

export function ImportSpanshDialog({ dialogOpen, onClose, }: ImportSpanshDialogProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const toast = useToast();
  const [processing, setProcessing] = useState(false);

  const routeState = useContext(RouteContext)!;

  useEffect(() => {
    const dialog = dialogRef.current;
    if (dialogOpen) {
      if (!dialog?.open) {
        dialog?.showModal();
      }
    } else {
      dialog?.close();
    }
  }, [dialogOpen]);

  useEffect(() => {
    if (!dialogOpen) {
      return;
    }

    const onPaste = async (event: ClipboardEvent) => {
      const htmlData = event.clipboardData?.getData("text/html");
      if (!htmlData) {
        return;
      }

      setProcessing(true);
      try {
        const res = await parseSpanshHtml(htmlData);
        routeState.setRoute(res);
      } catch (error) {
        console.error("Failed to parse Spansh HTML:", error);
        toast.showError((error as Error).message);
      } finally {
        setProcessing(false);
      }
    };

    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [dialogOpen, routeState, toast]);

  return (
    <dialog
      className={`${uiTheme.glassPanel} fixed left-1/2 top-1/2 z-999 m-0 w-[min(92vw,700px)] -translate-x-1/2 -translate-y-1/2 p-5`}
      onClose={onClose}
      ref={dialogRef}
    >
      <div className={`${uiTheme.panelHeader} mb-4`}>
        <h2 className={uiTheme.panelTitle}>Import route from Spansh</h2>
        <Button aria-label="Close import route dialog" onClick={onClose} variant="icon">
          ×
        </Button>
      </div>

      <div className="space-y-4 text-sm leading-6 text-white/88">
        <p>
          The built-in plotting system only considers neutron stars in the route. For some routes at the edges of the galaxy, there might not be enough neutron stars to make a good route. For these situations, using the Spansh galaxy plotter is recommended. It takes into account other types of stars as well.
        </p>
        <p>
          After generating the route in Spansh, you can import the route by selecting the HTML in the Spansh page with CTRL+A and pasting it into this page with CTRL+V
        </p>
        <p>
          Note that the Spansh Galaxy plotter requires you to import your ship's exact build from <a className="text-space-accent" href="https://coriolis.io/">coriolis.io</a> or <a className="text-space-accent" href="https://edsy.org/">edsy.org</a>
        </p>
      </div>

      <div className="mt-5 flex flex-wrap items-center justify-between gap-3 border-t border-white/20 pt-4">
        <a
          className="inline-flex items-center justify-center border border-space-accent-strong bg-space-accent-strong px-3 py-1.5 text-xs font-semibold text-white shadow-[0_2px_10px_rgba(47,127,243,0.5)] transition hover:bg-space-accent hover:shadow-[0_4px_16px_rgba(47,127,243,0.62)]"
          href="https://spansh.co.uk/exact-plotter"
          rel="noreferrer"
          target="_blank"
        >
          Open Spansh Galaxy Plotter
        </a>

        <div className={`flex items-center gap-2 text-sm ${processing ? "text-white/88" : "text-emerald-300"}`}>
          <span aria-hidden="true" className="inline-block h-2.5 w-2.5 rounded-full bg-emerald-400 shadow-[0_0_10px_rgba(74,222,128,0.75)] animate-pulse" />
          <span>{processing ? "Processing route..." : "Waiting for CTRL+V..."}</span>
        </div>
      </div>
    </dialog>
  );
}