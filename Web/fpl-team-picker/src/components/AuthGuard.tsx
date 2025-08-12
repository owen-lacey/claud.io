"use client";

import { useCallback, useState } from "react";
import { Dialog, DialogBackdrop, DialogPanel, DialogTitle } from '@headlessui/react'

function AuthGuard({ onDone }: { onDone: (cookie: string) => void }) {
  const [open, setOpen] = useState(true);
  const [value, setValue] = useState('');

  const onSubmit = useCallback(() => {
    onDone(value);
    setOpen(false);
  }, [onDone, value])


  return (
    <Dialog open={open} onClose={() => null} className="relative z-10">
      <DialogBackdrop
        transition
        className="fixed inset-0 bg-black/20 backdrop-blur-sm transition-opacity data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in"
      />

      <div className="fixed inset-0 z-10 w-screen overflow-y-auto">
        <div className="flex min-h-full items-end justify-center p-4 text-center sm:items-center sm:p-0">
          <DialogPanel
            transition
            className="relative transform rounded-xl bg-card text-card-foreground border border-border/50 shadow-2xl transition-all data-closed:translate-y-4 data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in sm:my-8 sm:w-full sm:max-w-lg data-closed:sm:translate-y-0 data-closed:sm:scale-95"
          >
            <div className="p-6">
              <div className="flex items-start">
                <div className="px-2 text-center overflow-auto text-left">
                  <DialogTitle as="h3" className="text-lg font-semibold text-foreground mb-4">
                    Welcome to Auto FPL! &#9917;
                  </DialogTitle>
                  <div className="text-muted-foreground leading-relaxed">
                    Auto FPL is a read-only view of your FPL profile. It uses optimisation algorithms to help you pick* the best FPL team.
                    <br />
                    <br />
                    <span className="text-foreground font-medium">Let&apos;s get started:</span>
                    <div className="flex flex-col gap-3 mt-4">
                      <div className="flex items-start gap-3">
                        <span className="w-7 h-7 bg-primary/10 text-primary text-sm font-semibold flex items-center justify-center rounded-full shrink-0 mt-0.5">1</span>
                        <p>Log in to <a className="text-primary hover:text-primary/80 transition-colors font-medium" href="https://fantasy.premierleague.com" target="_blank">fantasy.premierleague.com</a>.</p>
                      </div>
                      <div className="flex items-start gap-3">
                        <span className="w-7 h-7 bg-primary/10 text-primary text-sm font-semibold flex items-center justify-center rounded-full shrink-0 mt-0.5">2</span>
                        <p>Open dev tools (<span className="font-medium text-xs font-mono bg-muted px-2 py-1 rounded-md text-muted-foreground">F12</span>) and run the following:</p>
                      </div>
                      <pre className="py-4 px-4 text-xs bg-muted rounded-lg overflow-x-auto overflow-y-hidden font-mono text-muted-foreground border border-border/50 whitespace-nowrap">
localStorage.getItem(&apos;oidc.user:https://account.premierleague.com/as:bfcbaf69-aade-4c1b-8f00-c1cb8a193030&apos;).match(/&quot;access_token&quot;:&quot;([^&quot;]+)&quot;/)[1]
                      </pre>
                      <div className="flex items-start gap-3">
                        <span className="w-7 h-7 bg-primary/10 text-primary text-sm font-semibold flex items-center justify-center rounded-full shrink-0 mt-0.5">3</span>
                        <p>Paste the value (excluding single-quotes) here:</p>
                      </div>
                    </div>
                  </div>
                  <div className="mt-4 mb-1 flex w-full">
                    <input 
                      autoFocus 
                      type="text" 
                      className="flex-1 font-mono text-sm bg-background border border-input rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-ring focus:border-transparent" 
                      value={value} 
                      onChange={val => setValue(val.target.value)} 
                      placeholder="Paste your access token here..."
                    />
                  </div>
                </div>
              </div>
              <p className="text-xs px-2 text-muted-foreground mt-4 italic text-right">
                * probably
              </p>
            </div>
            <div className="bg-muted/30 px-6 py-4 flex flex-row-reverse border-t border-border/50">
              <button
                type="button"
                onClick={onSubmit}
                className="px-6 py-2.5 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-all duration-200 font-medium shadow-sm hover:shadow-md"
              >
                Let&apos;s go! &#128640;
              </button>
            </div>
          </DialogPanel>
        </div>
      </div>
    </Dialog>
  );
}

export default AuthGuard;