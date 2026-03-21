import type { RouteNode } from "./types";
import { uiTheme } from "./theme";

interface RouteListPanelProps {
  nodes: RouteNode[];
  visible: boolean;
  currentProgress: number;
  onSetProgress: (nextProgress: number) => void;
}

export function RouteListPanel({ currentProgress, nodes, onSetProgress, visible }: RouteListPanelProps) {
  if (!visible || nodes.length === 0) {
    return null;
  }

  return (
    <div className={`${uiTheme.glassPanel} fixed left-5 top-20 z-998 min-w-70 max-w-[calc(100vw-40px)] w-[min(380px,calc(100vw-40px))] min-h-55 h-[min(60vh,520px)] max-h-[80vh] resize overflow-hidden p-0`}>
      <div className="border-b border-white/15 px-4 py-3">
        <h3 className="m-0 text-sm font-semibold tracking-[0.03em] text-white/90">Plotted Route</h3>
      </div>

      <div className="h-[calc(100%-48px)] overflow-y-auto">
        <table className="w-full border-collapse text-sm text-white/85">
          <thead>
            <tr className="border-b border-white/10 bg-white/5 text-left text-[11px] uppercase tracking-[0.05em] text-white/65">
              <th className="w-10 px-3 py-2"></th>
              <th className="px-3 py-2">Star</th>
            </tr>
          </thead>
          <tbody>
            {nodes.map((node, index) => {
              const checkboxId = `route-node-${index}`;

              return (
                <tr className="border-b border-white/8 last:border-b-0" key={`${node.name}-${index}`}>
                  <td className="px-3 py-2">
                    <input
                      checked={index <= currentProgress}
                      className="h-3.5 w-3.5 cursor-pointer border border-white/35 bg-black/30 accent-space-accent-strong"
                      id={checkboxId}
                      onInput={(event) => {
                        const nextChecked = (event.currentTarget as HTMLInputElement).checked;
                        const nextIndex = nextChecked ? index : index - 1;
                        onSetProgress(nextIndex);
                      }}
                      type="checkbox"
                    />
                  </td>
                  <td className="px-3 py-2 font-medium text-white/90">
                    <label className="cursor-pointer" htmlFor={checkboxId}>{node.name}</label>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
