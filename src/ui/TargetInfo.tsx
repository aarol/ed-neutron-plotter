import type { TargetInfoState } from "./types";

interface TargetInfoProps {
  target: TargetInfoState;
  onOpenRoute: (targetName: string) => void;
}

export function TargetInfo({ onOpenRoute, target }: TargetInfoProps) {
  return (
    <div className="fixed bottom-4.5 left-1/2 z-1000 flex -translate-x-1/2 items-center gap-2.5 border border-white/25 bg-black/70 px-4 py-1.5 text-base tracking-[0.03em] whitespace-nowrap text-white/85 shadow-[0_4px_15px_rgba(0,0,0,0.3)] backdrop-blur-md">
      <span className="mr-1.5 font-semibold text-white">{target.name}</span>
      <span className="text-white/55 [font-variant-numeric:tabular-nums]">
        ({target.x.toFixed(2)}, {target.y.toFixed(2)}, {target.z.toFixed(2)})
      </span>
      <button
        aria-label="Plot route"
        className="ml-0.5 flex h-8 w-8 shrink-0 items-center justify-center border border-transparent bg-transparent p-0 text-white/70 transition hover:border-white/50 hover:bg-white/12 hover:text-white"
        onClick={() => onOpenRoute(target.name)}
        title="Find route to target"
        type="button"
      >
        <svg className="h-5 w-5" fill="currentColor" viewBox="0 -960 960 960">
          <path d="M320-360h80v-120h140v100l140-140-140-140v100H360q-17 0-28.5 11.5T320-520v160ZM480-80q-15 0-29.5-6T424-104L104-424q-12-12-18-26.5T80-480q0-15 6-29.5t18-26.5l320-320q12-12 26.5-18t29.5-6q15 0 29.5 6t26.5 18l320 320q12 12 18 26.5t6 29.5q0 15-6 29.5T856-424L536-104q-12 12-26.5 18T480-80ZM320-320l160 160 320-320-320-320-320 320 160 160Zm160-160Z" />
        </svg>
      </button>
    </div>
  );
}