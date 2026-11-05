import React, { createContext, useContext, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertCircle, CheckCircle2, Info, X } from 'lucide-react';
import { clsx } from 'clsx';

export type ToastType = 'info' | 'success' | 'warning' | 'error';

export interface Toast {
  id: string;
  type: ToastType;
  title: string;
  message?: string;
  durationMs?: number;
}

interface ToastContextValue {
  showToast: (toast: Omit<Toast, 'id'>) => void;
  removeToast: (id: string) => void;
}

const ToastContext = createContext<ToastContextValue | undefined>(undefined);

export const ToastProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const showToast = useCallback(
    ({ type, title, message, durationMs = 4000 }: Omit<Toast, 'id'>) => {
      const id = Math.random().toString(36).substring(2, 9);
      setToasts((prev) => [...prev, { id, type, title, message, durationMs }]);

      if (durationMs > 0) {
        setTimeout(() => {
          removeToast(id);
        }, durationMs);
      }
    },
    [removeToast]
  );

  const icons = {
    info: <Info className="w-4 h-4 text-primary flex-shrink-0" />,
    success: <CheckCircle2 className="w-4 h-4 text-genuine flex-shrink-0" />,
    warning: <AlertCircle className="w-4 h-4 text-uncertain flex-shrink-0" />,
    error: <AlertCircle className="w-4 h-4 text-spoof flex-shrink-0" />,
  };

  const borders = {
    info: 'border-primary/30',
    success: 'border-genuine/30',
    warning: 'border-uncertain/30',
    error: 'border-spoof/30',
  };

  return (
    <ToastContext.Provider value={{ showToast, removeToast }}>
      {children}
      <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 max-w-sm w-full pointer-events-none">
        <AnimatePresence>
          {toasts.map((toast) => (
            <motion.div
              key={toast.id}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 6 }}
              transition={{ duration: 0.15 }}
              className={clsx(
                'pointer-events-auto bg-surface border p-3.5 rounded-[10px] shadow-card flex items-start gap-3 relative overflow-hidden',
                borders[toast.type]
              )}
            >
              {icons[toast.type]}
              <div className="flex-1 min-w-0">
                <h4 className="text-xs font-semibold text-txt-primary">
                  {toast.title}
                </h4>
                {toast.message && (
                  <p className="text-xs text-txt-secondary mt-0.5 leading-snug">
                    {toast.message}
                  </p>
                )}
              </div>
              <button
                onClick={() => removeToast(toast.id)}
                className="text-txt-muted hover:text-txt-primary transition-colors duration-150 p-1"
                aria-label="Dismiss toast"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </ToastContext.Provider>
  );
};

export const useToast = () => {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error('useToast must be used within a ToastProvider');
  }
  return context;
};
