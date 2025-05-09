import { Dialog, DialogBackdrop, DialogPanel, Tab, TabGroup, TabList, TabPanel, TabPanels } from "@headlessui/react";
import { useEffect, useState } from "react";
import { FplApi } from "../../helpers/fpl-api";
import { MyTeam, Transfers } from "../../helpers/api";
import ShowTransfers from "./ShowTransfers";

function CalculateModal({ open, onClose }: { open: boolean, onClose: () => void }) {
  const [transfers, setTransfers] = useState<Transfers | null>(null);
  const [wildcard, setWildcard] = useState<MyTeam | null>(null);

  const calculateTransfers = async () => {
    const [transfersResult, wildcardResult] = await Promise.all([
      new FplApi().transfers.transfersCreate(),
      new FplApi().wildcard.wildcardCreate()
    ]);

    setTransfers(transfersResult.data);
    setWildcard(wildcardResult.data);
  }

  useEffect(() => {
    if (open) {
      calculateTransfers();
    }
  }, [open])

  return <Dialog open={open} onClose={() => null} className="relative z-10">
    <DialogBackdrop
      transition
      className="fixed inset-0 bg-gray-500/75 transition-opacity data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in"
    />

    <div className="fixed inset-0 z-10 w-screen overflow-y-auto">
      <div className="flex min-h-full items-end justify-center p-4 text-center sm:items-center sm:p-0">
        <DialogPanel
          transition
          className="relative transform overflow-hidden rounded-lg bg-white text-left shadow-xl transition-all data-closed:translate-y-4 data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in sm:my-8 data-closed:sm:translate-y-0 data-closed:sm:scale-95"
        >
          {transfers && wildcard ? <div className="px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
            <TabGroup>
              <TabList className="flex gap-4">
                <Tab
                  className="rounded-md py-1 px-3 text-sm/6 font-semibold focus:outline-none data-[selected]:bg-gray-200 hover:bg-gray-200"
                >
                  <div>Transfers</div>
                </Tab>
                <Tab
                  className="rounded-md py-1 px-3 text-sm/6 font-semibold focus:outline-none data-[selected]:bg-gray-200 hover:bg-gray-200"
                >
                  <div>Wildcard</div>
                </Tab>
              </TabList>
              <TabPanels className="mt-3">
                <TabPanel className="p-0 rounded-xl bg-white/5">
                  <ShowTransfers transferPenaliseCount={Math.min(transfers!.freeTransfers!, 0)} startingXi={transfers.startingXi!} bench={transfers.bench!} bank={transfers.bank!} />
                </TabPanel>
                <TabPanel className="p-0 rounded-xl bg-white/5">
                  <ShowTransfers transferPenaliseCount={0} startingXi={wildcard.selectedSquad!.startingXi!} bench={wildcard.selectedSquad!.bench!} bank={wildcard.bank!} />
                </TabPanel>
              </TabPanels>
            </TabGroup>
          </div>
            : <></>}
          <div className="bg-gray-50 px-4 py-3 sm:flex sm:flex-row-reverse sm:px-6">
            <button
              type="button"
              onClick={onClose}
              className="info inline-flex justify-center px-3 py-2 w-full sm:ml-3 sm:w-auto"
            >
              Close
            </button>
          </div>
        </DialogPanel>
      </div>
    </div>
  </Dialog>;
}

export default CalculateModal;
