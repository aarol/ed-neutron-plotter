import { useEffect, useRef, useState } from "preact/hooks";
import type { ComponentChildren } from "preact";

export type PanelPosition = {
  x: number;
  y: number;
};

interface DraggablePanelProps {
  children: (props: { isDragging: boolean }) => ComponentChildren;
  initialPosition?: PanelPosition;
  className?: string;
  dragHandleClass?: string;
  onPositionChange?: (position: PanelPosition) => void;
}

export function DraggablePanel({
  children,
  initialPosition = { x: 20, y: 80 },
  className = "",
  dragHandleClass = "drag-handle",
  onPositionChange,
}: DraggablePanelProps) {
  const panelRef = useRef<HTMLDivElement>(null);
  const dragStateRef = useRef<{ pointerId: number; offsetX: number; offsetY: number } | null>(null);
  const [position, setPosition] = useState<PanelPosition>(initialPosition);
  const [isDragging, setIsDragging] = useState(false);

  const handlePointerDown = (event: PointerEvent) => {
    if (event.pointerType === "mouse" && event.button !== 0) {
      return;
    }

    const dragHandle = (event.target as Element).closest(`.${dragHandleClass}`);
    if (!dragHandle) {
      return;
    }

    const panel = panelRef.current;
    if (!panel) {
      return;
    }

    const rect = panel.getBoundingClientRect();
    dragStateRef.current = {
      pointerId: event.pointerId,
      offsetX: event.clientX - rect.left,
      offsetY: event.clientY - rect.top,
    };
    setIsDragging(true);
  };

  useEffect(() => {
    if (!isDragging) {
      return;
    }

    const handlePointerMove = (event: PointerEvent) => {
      const dragState = dragStateRef.current;
      const panel = panelRef.current;
      if (!dragState || !panel || dragState.pointerId !== event.pointerId) {
        return;
      }

      const panelWidth = panel.offsetWidth;
      const panelHeight = panel.offsetHeight;

      const nextX = Math.min(
        Math.max(0, event.clientX - dragState.offsetX),
        Math.max(0, window.innerWidth - panelWidth),
      );
      const nextY = Math.min(
        Math.max(0, event.clientY - dragState.offsetY),
        Math.max(0, window.innerHeight - panelHeight),
      );

      const nextPosition = { x: nextX, y: nextY };
      setPosition(nextPosition);
      onPositionChange?.(nextPosition);
    };

    const handlePointerStop = (event: PointerEvent) => {
      const dragState = dragStateRef.current;
      if (!dragState || dragState.pointerId !== event.pointerId) {
        return;
      }

      dragStateRef.current = null;
      setIsDragging(false);
    };

    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerStop);
    window.addEventListener("pointercancel", handlePointerStop);

    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerStop);
      window.removeEventListener("pointercancel", handlePointerStop);
    };
  }, [isDragging, onPositionChange]);

  return (
    <div
      className={className}
      ref={panelRef}
      style={{ left: `${position.x}px`, top: `${position.y}px` }}
      onPointerDown={handlePointerDown}
    >
      {children({ isDragging })}
    </div>
  );
}
