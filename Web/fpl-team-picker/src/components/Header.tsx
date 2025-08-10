"use client";

import React, { memo, useCallback, useContext } from "react";
import { ArrowPathRoundedSquareIcon } from '@heroicons/react/24/solid';
import { Popover, PopoverButton, PopoverPanel } from "@headlessui/react";
import Link from "next/link";
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
  const assistantEnabled = process.env.NEXT_PUBLIC_FPL_ASSISTANT_ENABLED === 'true';

  return <header className="bg-card border border-border shadow-lg rounded-lg py-4 px-6 flex justify-between items-center">
    <h1 className="text-2xl font-semibold text-foreground">Welcome, <span>{myDetails.output!.firstName}</span>!</h1>
    <div className="text-sm flex items-center gap-3 text-foreground">
      {assistantEnabled && (
        <Link href="/assistant" className="underline underline-offset-4 hover:text-accent-foreground">
          Assistant (beta)
        </Link>
      )}
      <div className="flex items-center">
        <span>User ID: &nbsp;</span>

        <Popover className="relative">
          <PopoverButton className="font-normal font-mono bg-muted text-muted-foreground px-2 py-1 rounded-md focus:outline-none hover:bg-accent hover:text-accent-foreground">{myDetails.output!.id}</PopoverButton>

          <PopoverPanel
            transition
            anchor="bottom"
            className="z-8 rounded-md border border-border bg-popover text-popover-foreground text-sm/6 transition duration-200 ease-in-out data-[closed]:-translate-y-1 data-[closed]:opacity-0">
            <button className="flex gap-2 p-4 hover:bg-accent hover:text-accent-foreground transition-colors" onClick={clearLocal}>
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
