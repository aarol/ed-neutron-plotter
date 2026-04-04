import { Button } from "./components/Button";
import { DraggablePanel } from "./components/DraggablePanel";
import { uiTheme } from "./theme";
import { useContext } from "preact/hooks";
import { RouteContext } from "./state/routeModel";
import { For } from "@preact/signals/utils";


export function RouteListPanel() {
  const { nodes, progress, setProgress, clearRoute } = useContext(RouteContext)!;

  return (
    <DraggablePanel
      className={`${uiTheme.glassPanel} fixed z-998 min-w-100 max-w-[calc(100vw-40px)] w-[min(380px,calc(100vw-40px))] min-h-60 h-[min(60vh,520px)] max-h-[80vh] resize overflow-hidden p-0`}
      initialPosition={{ x: 20, y: 80 }}
    >
      {({ isDragging }) => (
        <>
          <div
            className={`drag-handle flex items-center justify-between border-b border-white/15 px-4 py-3 select-none touch-none ${isDragging ? "cursor-grabbing" : "cursor-grab"}`}
          >
            <h3 className="m-0 text-sm font-semibold tracking-[0.03em] text-white/90">Plotted Route</h3>
            <Button aria-label="Close plotted route" className="h-7 w-7 rounded" onClick={clearRoute} variant="icon">
              x
            </Button>
          </div>

          <div className="h-[calc(100%-48px)] overflow-auto pb-3">
            <table className="w-full border-collapse text-sm text-white/85">
              <thead>
                <tr className="border-b border-white/10 bg-white/5 text-left text-nowrap text-[11px] uppercase tracking-[0.05em] text-white/65">
                  <th className="pl-3 w-0 py-2">Jump #</th>
                  <th className="w-10 px-3 py-2"></th>
                  <th className="px-3 py-2">System</th>
                  <th className="w-26 px-3 py-2">Distance (LY)</th>
                  <th className="w-20 px-3 py-2">Flags</th>
                </tr>
              </thead>
              <tbody>
                <For each={nodes}>
                  {(node, index) => {
                    const checkboxId = `route-node-${index}`;

                    return (
                      <tr className="border-b border-white/8 last:border-b-0" key={`${node.system.name}-${index}`}>
                        <td className="px-3 py-2 tabular-nums text-white/70">{index + 1}</td>
                        <td className="px-3 py-2">
                          <input
                            checked={index < progress.value}
                            className="h-3.5 w-3.5 cursor-pointer border border-white/35 bg-black/30 accent-space-accent-strong"
                            id={checkboxId}
                            onInput={(event) => {
                              const nextChecked = (event.currentTarget as HTMLInputElement).checked;
                              const nextIndex = nextChecked ? index + 1 : index;
                              setProgress(nextIndex);
                            }}
                            type="checkbox"
                          />
                        </td>
                        <td className="font-medium text-white/90">
                          <label className="cursor-pointer" htmlFor={checkboxId}>{node.system.name}</label>
                        </td>
                        <td className="px-3 py-2 tabular-nums text-white/80">{node.distance.toFixed(2)}</td>
                        <td className="px-3 py-2 text-white/80">
                          <div className="flex items-center gap-2 font-semibold">
                            {node.refuel ? (
                              <span className="text-orange-400" title="Refuel at this star">R</span>
                            ) : (
                              <span className="text-white/35">-</span>
                            )}
                            {node.isNeutron ? (
                              <span className="text-sky-300" title="Neutron star">N</span>
                            ) : (
                              <span className="text-white/35">-</span>
                            )}
                          </div>
                        </td>
                      </tr>
                    );
                  }}
                </For>
              </tbody>
            </table>
          </div>
        </>
      )}
    </DraggablePanel>
  );
}
