import { memo, useCallback, useContext } from "react";
import { DataContext } from "../App.tsx";
import { ArrowPathRoundedSquareIcon } from '@heroicons/react/24/solid';
import { Popover, PopoverButton, PopoverPanel } from "@headlessui/react";
import { useLocalStorage } from "@uidotdev/usehooks";

const Header = memo(function Header() {
  const allData = useContext(DataContext);
  const [_, savePlProfile] = useLocalStorage<string | null>("pl_profile", null);

  const clearLocal = useCallback(() => {
    savePlProfile(null);
  }, [])

  if (!allData?.myDetails) {
    return <></>
  }
  const { myDetails } = allData;

  return <header className="bg-white border border-gray-300 shadow-lg rounded-lg py-4 px-6 flex justify-between items-center">
    <h1 className="text-2xl font-semibold">Welcome, <span>{myDetails.firstName}</span>!</h1>
    <div className="text-sm flex items-center">
      <span>User ID: &nbsp;</span>

      <Popover className="relative">

        <PopoverButton className="font-normal font-mono bg-gray-100 px-2 py-1 rounded-md focus:outline-none">{myDetails.id}</PopoverButton>


        <PopoverPanel
          transition
          anchor="bottom"
          className="z-8 rounded-md border border-gray-200 bg-white text-sm/6 transition duration-200 ease-in-out data-[closed]:-translate-y-1 data-[closed]:opacity-0">
          <button className="flex gap-2 p-4" onClick={clearLocal}>
            <ArrowPathRoundedSquareIcon className="w-4" />
            Reset
          </button>
        </PopoverPanel>
      </Popover>
    </div>
  </header>

});

export default Header;
