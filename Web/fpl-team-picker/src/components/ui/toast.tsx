"use client";

import { useEffect, useState } from "react";
import { CheckCircle, AlertCircle, X } from "lucide-react";

export type ToastType = "success" | "error" | "info";

export type Toast = {
  id: string;
  type: ToastType;
  message: string;
  duration?: number;
};

type ToastManagerState = {
  toasts: Toast[];
};

class ToastManager {
  private listeners: ((state: ToastManagerState) => void)[] = [];
  private state: ToastManagerState = { toasts: [] };

  subscribe(listener: (state: ToastManagerState) => void) {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter(l => l !== listener);
    };
  }

  private emit() {
    this.listeners.forEach(listener => listener(this.state));
  }

  show(message: string, type: ToastType = "info", duration = 3000) {
    const id = Math.random().toString(36).substr(2, 9);
    const toast: Toast = { id, message, type, duration };
    
    this.state.toasts.push(toast);
    this.emit();

    if (duration > 0) {
      setTimeout(() => {
        this.dismiss(id);
      }, duration);
    }
  }

  dismiss(id: string) {
    this.state.toasts = this.state.toasts.filter(t => t.id !== id);
    this.emit();
  }

  success(message: string, duration?: number) {
    this.show(message, "success", duration);
  }

  error(message: string, duration?: number) {
    this.show(message, "error", duration);
  }

  info(message: string, duration?: number) {
    this.show(message, "info", duration);
  }
}

export const toastManager = new ToastManager();

export function useToasts() {
  const [state, setState] = useState<ToastManagerState>({ toasts: [] });

  useEffect(() => {
    return toastManager.subscribe(setState);
  }, []);

  return {
    toasts: state.toasts,
    show: toastManager.show.bind(toastManager),
    dismiss: toastManager.dismiss.bind(toastManager),
    success: toastManager.success.bind(toastManager),
    error: toastManager.error.bind(toastManager),
    info: toastManager.info.bind(toastManager),
  };
}

export function ToastContainer() {
  const { toasts, dismiss } = useToasts();

  if (toasts.length === 0) return null;

  return (
    <div className="fixed top-4 right-4 z-50 flex flex-col gap-2">
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} onDismiss={() => dismiss(toast.id)} />
      ))}
    </div>
  );
}

function ToastItem({ toast, onDismiss }: { toast: Toast; onDismiss: () => void }) {
  const icon = {
    success: <CheckCircle className="h-4 w-4 text-green-600" />,
    error: <AlertCircle className="h-4 w-4 text-red-600" />,
    info: <AlertCircle className="h-4 w-4 text-blue-600" />,
  }[toast.type];

  const bgColor = {
    success: "bg-green-50 border-green-200 dark:bg-green-900/30 dark:border-green-800",
    error: "bg-red-50 border-red-200 dark:bg-red-900/30 dark:border-red-800", 
    info: "bg-blue-50 border-blue-200 dark:bg-blue-900/30 dark:border-blue-800",
  }[toast.type];

  return (
    <div className={`${bgColor} border rounded-md shadow-lg p-3 min-w-64 max-w-sm animate-in slide-in-from-right duration-300`}>
      <div className="flex items-start gap-2">
        {icon}
        <div className="flex-1 text-sm">{toast.message}</div>
        <button
          onClick={onDismiss}
          className="text-muted-foreground hover:text-foreground transition-colors"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}
