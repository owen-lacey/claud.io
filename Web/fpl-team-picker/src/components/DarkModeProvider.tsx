"use client";

import { useEffect } from 'react';

export default function DarkModeProvider({ children }: { children: React.ReactNode }) {
  useEffect(() => {
    // Always set dark mode
    document.documentElement.classList.add('dark');
    
    if (typeof window !== 'undefined') {
      localStorage.setItem('theme', 'dark');
    }
  }, []);

  return <>{children}</>;
}
