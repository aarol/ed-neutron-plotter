import type { TargetInfoState } from "./types";
import { uiTheme } from "./theme";

interface TargetInfoProps {
  target: TargetInfoState;
  onOpenRoute: (targetName: string) => void;
}

export function TargetInfo({ onOpenRoute, target }: TargetInfoProps) {
  return (
    <div className={`${uiTheme.glassPanel} fixed bottom-8 left-1/2 z-1000 flex -translate-x-1/2 items-stretch text-lg tracking-[0.03em] whitespace-nowrap text-white/85`}>
      <div className="flex items-center gap-2.5 px-4 py-1.5">
        <span className="mr-1.5 font-semibold text-white">{target.name}</span>
        <span className="text-sm text-white/55 [font-variant-numeric:tabular-nums]">
          ({target.x.toFixed(2)}, {target.y.toFixed(2)}, {target.z.toFixed(2)})
        </span>
      </div>

      <div className="flex items-stretch border-l border-white/35">
        <button
          aria-label="Plot route"
          className="flex h-full shrink-0 items-center justify-center px-3 text-lg font-semibold tracking-[0.04em] text-white/80 transition hover:bg-white/10 hover:text-white"
          onClick={() => onOpenRoute(target.name)}
          title="Find route to target"
          type="button"
        >
          Plot
        </button>
      </div>
    </div>
  );
}