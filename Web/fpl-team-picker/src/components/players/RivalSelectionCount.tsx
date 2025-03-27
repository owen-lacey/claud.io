import { Popover, PopoverButton, PopoverPanel } from "@headlessui/react";
import { RivalTeam } from "../../models/rival-league";

function RivalSelectionCount({ playerId, rivalTeams }: { playerId: number, rivalTeams: RivalTeam[] }) {

  const selections = rivalTeams.filter(rt => {
    return rt.team.startingXi?.some(p => p.player!.id === playerId) || rt.team.bench?.some(p => p.player!.id === playerId);
  })

  const totalRivals = rivalTeams.length;
  const right = ((1 - selections.length / totalRivals) * 100).toFixed(0);

  return <Popover className="relative">
    <PopoverButton className="selection-counter focus:outline-none ">
      <div className="bg-linear-to-r from-cyan-600 to-blue-900 rounded" style={{ right: `${right}%` }}></div>
    </PopoverButton>
    <PopoverPanel
      transition
      anchor="bottom"
      className="z-8 rounded-md border border-gray-200 bg-white text-sm/6 transition duration-200 ease-in-out data-[closed]:-translate-y-1 data-[closed]:opacity-0"
    >
      <div className="p-3">
        {selections.length > 0
          ? <div className="flex flex-col">
            <div className="font-semibold">Selected by {(selections.length / totalRivals * 100).toFixed(0)}%</div>
            {selections.sort((a, b) => a.rival!.position! - b.rival!.position!).map((s, i) =>
              <div className="flex justify-between gap-2" key={i}>
                <div className="">{s.rival.playerName}</div>
                <div className="font-mono font-xs">{s.rival.position}</div>
              </div>)}
          </div>
          : 'Player not selected'}
      </div>
    </PopoverPanel>
  </Popover>
}

export default RivalSelectionCount;