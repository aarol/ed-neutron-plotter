import { createContext, type ComponentChildren } from "preact";
import { useCallback, useContext, useMemo, useRef, useState } from "preact/hooks";

interface ToastItem {
  id: number;
  message: string;
}

interface ToastContextValue {
  showError: (message: string) => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

export function useToast(): ToastContextValue {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error("useToast must be used within ToastProvider");
  }
  return context;
}

export function ToastProvider({ children }: { children: ComponentChildren }) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const toastIdRef = useRef(0);
  const toastTimeoutsRef = useRef(new Map<number, number>());

  const dismissToast = useCallback((id: number) => {
    setToasts((currentToasts) => currentToasts.filter((toast) => toast.id !== id));
    const timeoutId = toastTimeoutsRef.current.get(id);
    if (timeoutId !== undefined) {
      window.clearTimeout(timeoutId);
      toastTimeoutsRef.current.delete(id);
    }
  }, []);

  const showError = useCallback((message: string) => {
    const id = ++toastIdRef.current;
    setToasts((currentToasts) => [...currentToasts, { id, message }]);
    const timeoutId = window.setTimeout(() => {
      dismissToast(id);
    }, 5500);
    toastTimeoutsRef.current.set(id, timeoutId);
  }, [dismissToast]);

  const contextValue = useMemo<ToastContextValue>(() => ({ showError }), [showError]);

  return (
    <ToastContext.Provider value={contextValue}>
      {children}
      <div className="pointer-events-none fixed bottom-4 right-4 z-90 flex max-w-[min(92vw,420px)] flex-col gap-2">
        {toasts.map((toast) => (
          <div
            className="pointer-events-auto border border-red-300/40 bg-[#2a1014]/95 px-3.5 py-2.5 text-[12px] text-red-100 shadow-[0_8px_24px_rgba(0,0,0,0.45)]"
            key={toast.id}
            role="alert"
          >
            <div className="flex items-start gap-2.5">
              <span className="mt-px text-[11px] leading-none text-red-300">ERROR</span>
              <p className="m-0 flex-1 leading-relaxed">{toast.message}</p>
              <button
                className="border border-red-300/35 px-1 text-[11px] text-red-100/90 transition hover:bg-red-200/15"
                onClick={() => dismissToast(toast.id)}
                type="button"
              >
                x
              </button>
            </div>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}