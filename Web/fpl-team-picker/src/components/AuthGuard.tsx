import { useCallback, useState } from "react";
import { Dialog, DialogBackdrop, DialogPanel, DialogTitle } from '@headlessui/react'

function AuthGuard({ onDone }: { onDone: (cookie: string) => void }) {
  const [open, setOpen] = useState(true);
  const [value, setValue] = useState('');

  const onSubmit = useCallback(() => {
    onDone(value);
    setOpen(false);
  }, [value])


  return (
    <Dialog open={open} onClose={() => null} className="relative z-10">
      <DialogBackdrop
        transition
        className="fixed inset-0 bg-gray-500/75 transition-opacity data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in"
      />

      <div className="fixed inset-0 z-10 w-screen overflow-y-auto">
        <div className="flex min-h-full items-end justify-center p-4 text-center sm:items-center sm:p-0">
          <DialogPanel
            transition
            className="relative transform overflow-hidden rounded-lg bg-white text-left shadow-xl transition-all data-closed:translate-y-4 data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in sm:my-8 sm:w-full sm:max-w-lg data-closed:sm:translate-y-0 data-closed:sm:scale-95"
          >
            <div className="p-4">
              <div className="flex items-start">
                <div className="px-2 text-center min-w-100 text-left">
                  <DialogTitle as="h3" className="text-base font-semibold text-gray-900 mb-3">
                    Welcome to Auto FPL! &#9917;
                  </DialogTitle>
                  <div>
                    Auto FPL is a read-only view of your FPL profile. It uses optimisation algorithms to help you pick* the best FPL team.
                    <br />
                    <br />
                    Let's get started:
                    <div className="flex flex-col gap-2 mt-2">
                      <div className="flex">
                        <span className="w-6 h-6 bg-blue-100 text-blue-800 text-xs font-medium me-2 px-2.5 py-0.5 rounded-full flex items-center justify-center">1</span>
                        <p>Log in to <a className="text-blue-500" href="https://fantasy.premierleague.com" target="_blank">fantasy.premierleague.com</a>.</p>
                      </div>
                      <div className="flex">
                        <span className="w-6 h-6 bg-blue-100 text-blue-800 text-xs font-medium me-2 px-2.5 py-0.5 rounded-full flex items-center justify-center">2</span>
                        <p>Open dev tools (<span className="font-normal text-xs font-mono bg-gray-100 p-1 rounded-md">F12</span>) and run the following:</p>
                      </div>
                      <pre className="py-4 px-2 text-xs bg-gray-100 p-1 rounded-md overflow-auto">
                        document.cookie.match(/pl_profile=(?&lt;plProfileValue&gt;[^;]+)/).groups.plProfileValue
                      </pre>
                      <div className="flex">
                        <span className="w-6 h-6 bg-blue-100 text-blue-800 text-xs font-medium me-2 px-2.5 py-0.5 rounded-full flex items-center justify-center">3</span>
                        <p>Paste the value (excluding single-quotes) here:</p>
                      </div>
                    </div>
                  </div>
                  <div className="mt-2 flex w-100%">
                    <input autoFocus type="text" className="flex-1 font-mono text-xs" value={value} onChange={val => setValue(val.target.value)} />
                  </div>
                </div>
              </div>
              <i className="text-xs px-2 text-gray-500">
                * probably
              </i>
            </div>
            <div className="bg-gray-50 px-4 py-3 flex flex-row-reverse px-6">
              <button
                type="button"
                onClick={onSubmit}
                className="border text-sm border-gray-200 p-2 bg-linear-to-r from-[rgb(10,229,255)] to-[rgb(66,162,255)]"
              >
                Let's go! &#128640;
              </button>
            </div>
          </DialogPanel>
        </div>
      </div>
    </Dialog>
  );
}

export default AuthGuard;