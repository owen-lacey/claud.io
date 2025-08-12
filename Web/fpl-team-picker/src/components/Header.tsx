"use client";

import React, { memo, useCallback, useContext } from "react";
import { ArrowPathRoundedSquareIcon } from '@heroicons/react/24/solid';
import { Popover, PopoverButton, PopoverPanel } from "@headlessui/react";
import { DataContext } from "@/lib/contexts";
import { LoadingCard } from "./utils/Loading";

const Header = memo(function Header() {
  const allData = useContext(DataContext);

  const clearLocal = useCallback(() => {
    if (typeof window !== 'undefined') {
      localStorage.removeItem("pl_profile");
      // Force page reload to restart the app
      window.location.reload();
    }
  }, []);

  if (!allData?.myDetails.output) {
    return <LoadingCard />;
  }
  
  const { myDetails } = allData;

  return <header className="bg-card border border-border/50 shadow-sm rounded-xl py-5 px-7 flex justify-between items-center">
    <h1 className="text-2xl font-semibold text-foreground">Welcome, <span className="text-primary">{myDetails.output!.firstName}</span>!</h1>
    <div className="text-sm flex items-center gap-3 text-muted-foreground">
      <div className="flex items-center">
        <span>User ID: &nbsp;</span>

        <Popover className="relative">
          <PopoverButton className="font-medium font-mono bg-muted text-muted-foreground px-3 py-1.5 rounded-lg focus:outline-none hover:bg-accent hover:text-accent-foreground transition-colors">{myDetails.output!.id}</PopoverButton>

          <PopoverPanel
            transition
            anchor="bottom"
            className="z-8 rounded-lg border border-border/50 bg-popover text-popover-foreground text-sm shadow-lg transition duration-200 ease-in-out data-[closed]:-translate-y-1 data-[closed]:opacity-0">
            <button className="flex gap-2 p-4 hover:bg-accent hover:text-accent-foreground transition-colors rounded-lg w-full text-left" onClick={clearLocal}>
              <ArrowPathRoundedSquareIcon className="w-4" />
              Reset
            </button>
          </PopoverPanel>
        </Popover>
      </div>
    </div>
  </header>
});

export default Header;
