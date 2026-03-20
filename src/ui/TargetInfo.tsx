import type { TargetInfoState } from "./types";
import { uiTheme } from "./theme";

interface TargetInfoProps {
  target: TargetInfoState;
  onOpenRoute: (targetName: string) => void;
  isRoutePanelOpen: boolean;
}

export function TargetInfo({ isRoutePanelOpen, onOpenRoute, target }: TargetInfoProps) {
  const dividerClassName = isRoutePanelOpen ? "border-l border-white/35" : "border-l border-emerald-400/55";
  const plotButtonClassName = isRoutePanelOpen
    ? "flex h-full shrink-0 cursor-pointer items-center justify-center border border-white/35 bg-white/8 px-3 text-lg font-semibold tracking-[0.04em] text-white/80 shadow-none transition hover:border-white/55 hover:bg-white/14 hover:text-white"
    : "flex h-full shrink-0 cursor-pointer items-center justify-center border border-emerald-400/45 bg-emerald-400/8 px-3 text-lg font-semibold tracking-[0.04em] text-emerald-200 shadow-[0_0_12px_rgba(74,222,128,0.22)] transition hover:border-emerald-300/70 hover:bg-emerald-400/16 hover:text-emerald-100 hover:shadow-[0_0_16px_rgba(74,222,128,0.38)]";

  return (
    <div className={`${uiTheme.glassPanel} fixed bottom-8 left-1/2 z-1000 flex -translate-x-1/2 items-stretch text-lg tracking-[0.03em] whitespace-nowrap text-white/85`}>
      <div className="flex items-center gap-2.5 px-4 py-1.5">
        <span className="mr-1.5 font-semibold text-white">{target.name}</span>
        <span className="text-sm text-white/55 [font-variant-numeric:tabular-nums]">
          ({target.x.toFixed(2)}, {target.y.toFixed(2)}, {target.z.toFixed(2)})
        </span>
      </div>

      <div className={`flex items-stretch ${dividerClassName}`}>
        <button
          aria-label="Plot route"
          className={plotButtonClassName}
          onClick={() => onOpenRoute(target.name)}
          title="Find route to target"
          type="button"
        >
          PLOT
        </button>
      </div>
    </div>
  );
}