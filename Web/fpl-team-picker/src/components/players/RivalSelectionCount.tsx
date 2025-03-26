import { Popover, PopoverButton, PopoverPanel } from "@headlessui/react";
import { PlayerSelection } from "../../models/player-selection";

function RivalSelectionCount({ selections, totalRivals }: { selections: PlayerSelection[], totalRivals: number }) {

    const right = ((1 - selections.length / totalRivals) * 100).toFixed(0);

    return <Popover className="relative">
        <PopoverButton className="selection-counter focus:outline-none ">
            <div className="bg-linear-to-r from-cyan-500 to-blue-500 rounded" style={{ right: `${right}%` }}></div>
        </PopoverButton>
        <PopoverPanel
            transition
            anchor="bottom"
            className="z-8 rounded-md border border-gray-200 bg-white text-sm/6 transition duration-200 ease-in-out [--anchor-gap:var(--spacing-5)] data-[closed]:-translate-y-1 data-[closed]:opacity-0"
        >
            <div className="p-3">
                {selections.length > 0
                    ? <div className="flex flex-col">
                        <div className="font-semibold">Selected by {(selections.length / totalRivals * 100).toFixed(0)}%</div>
                        {selections.sort((a, b) => a.position - b.position).map((s, i) =>
                            <div className="flex justify-between gap-2" key={i}>
                                <div className="">{s.selectedBy}</div>
                                <div className="font-mono font-xs">{s.position}</div>
                            </div>)}
                    </div>
                    : 'Player not selected'}
            </div>
        </PopoverPanel>
    </Popover>
}

export default RivalSelectionCount;