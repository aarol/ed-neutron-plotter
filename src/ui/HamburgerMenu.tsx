import { useEffect, useRef, useState } from "preact/hooks";
import { Button } from "./components/Button";

interface HamburgerMenuProps {
  onImportRouteFromSpansh: () => void | Promise<void>;
}

export function HamburgerMenu({ onImportRouteFromSpansh }: HamburgerMenuProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [isImporting, setIsImporting] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handlePointerDown = (event: PointerEvent) => {
      const container = containerRef.current;
      if (container && !container.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener("pointerdown", handlePointerDown);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
    };
  }, []);

  const handleImportClick = async () => {
    if (isImporting) {
      return;
    }

    setIsImporting(true);
    try {
      if (onImportRouteFromSpansh) {
        await onImportRouteFromSpansh();
      }
      setIsOpen(false);
    } finally {
      setIsImporting(false);
    }
  };

  return (
    <div className="pointer-events-auto fixed right-5 top-5 z-1001" ref={containerRef}>
      <Button
        aria-expanded={isOpen}
        aria-haspopup="menu"
        aria-label="Open menu"
        className="flex h-11 w-11 items-center justify-center border border-white/35 bg-black/70 text-white shadow-[0_10px_30px_rgba(0,0,0,0.55)] backdrop-blur-xl transition hover:border-white/55 hover:bg-black/80"
        onClick={() => setIsOpen((previous) => !previous)}
        variant="plain"
      >
        {isOpen ? (
          <svg className="h-5 w-5" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2.4" viewBox="0 0 24 24">
            <line x1="6" x2="18" y1="6" y2="18" />
            <line x1="6" x2="18" y1="18" y2="6" />
          </svg>
        ) : (
          <svg className="h-5 w-5" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" viewBox="0 0 24 24">
            <line x1="4" x2="20" y1="7" y2="7" />
            <line x1="4" x2="20" y1="12" y2="12" />
            <line x1="4" x2="20" y1="17" y2="17" />
          </svg>
        )}
      </Button>

      {isOpen && (
        <div className="absolute right-0 top-[calc(100%+8px)] min-w-55 border border-white/30 bg-black/85 p-2 text-white shadow-[0_12px_36px_rgba(0,0,0,0.6)] backdrop-blur-xl">
          <Button
            className="w-full border border-transparent bg-white/8 px-3 py-2 text-left text-sm text-white transition hover:bg-white/15 disabled:cursor-not-allowed disabled:opacity-60"
            disabled={isImporting}
            onClick={() => {
              void handleImportClick();
            }}
            variant="plain"
          >
            Import route from spansh
          </Button>
        </div>
      )}
    </div>
  );
}