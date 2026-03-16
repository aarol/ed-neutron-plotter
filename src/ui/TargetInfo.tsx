import type { TargetInfoState } from "./types";
import styles from "./UI.module.css";

interface TargetInfoProps {
  target: TargetInfoState;
  onOpenRoute: (targetName: string) => void;
}

export function TargetInfo({ onOpenRoute, target }: TargetInfoProps) {
  return (
    <div className={styles.targetInfo}>
      <span className={styles.targetName}>{target.name}</span>
      <span className={styles.targetCoords}>
        ({target.x.toFixed(2)}, {target.y.toFixed(2)}, {target.z.toFixed(2)})
      </span>
      <button
        aria-label="Plot route"
        className={styles.targetRouteBtn}
        onClick={() => onOpenRoute(target.name)}
        title="Find route to target"
        type="button"
      >
        <svg fill="currentColor" viewBox="0 -960 960 960">
          <path d="M320-360h80v-120h140v100l140-140-140-140v100H360q-17 0-28.5 11.5T320-520v160ZM480-80q-15 0-29.5-6T424-104L104-424q-12-12-18-26.5T80-480q0-15 6-29.5t18-26.5l320-320q12-12 26.5-18t29.5-6q15 0 29.5 6t26.5 18l320 320q12 12 18 26.5t6 29.5q0 15-6 29.5T856-424L536-104q-12 12-26.5 18T480-80ZM320-320l160 160 320-320-320-320-320 320 160 160Zm160-160Z" />
        </svg>
      </button>
    </div>
  );
}