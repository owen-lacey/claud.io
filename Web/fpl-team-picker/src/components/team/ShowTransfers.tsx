import { SelectedPlayer } from "../../helpers/api";
import { ChevronRightIcon } from '@heroicons/react/24/solid';
import { playerBg } from "../../helpers/styles";
import { lookupTeam } from "../../helpers/lookups";
import { useContext } from "react";
import { DataContext } from "../../App";

function ShowTransfers({ startingXi, bench, transferPenaliseCount, bank }: { startingXi: SelectedPlayer[], bench: SelectedPlayer[], transferPenaliseCount: number, bank: number }) {
  const allData = useContext(DataContext);
  const existingStartingXi = allData!.myTeam!.output!.selectedSquad!.startingXi!;
  const existingBench = allData!.myTeam!.output!.selectedSquad!.bench!;
  const playersIn = [
    ...startingXi.filter(p => existingStartingXi.every(ep => ep.player?.id != p.player?.id) && existingBench.every(ep => ep.player?.id != p.player?.id)),
    ...bench.filter(p => existingStartingXi.every(ep => ep.player?.id != p.player?.id) && existingBench.every(ep => ep.player?.id != p.player?.id))
  ];
  const playersOut = [
    ...existingStartingXi.filter(p => startingXi.every(ep => ep.player?.id != p.player?.id) && bench.every(ep => ep.player?.id != p.player?.id)),
    ...existingBench.filter(p => startingXi.every(ep => ep.player?.id != p.player?.id) && bench.every(ep => ep.player?.id != p.player?.id))
  ];
  const transferCost = transferPenaliseCount * -4;
  const transferCount = playersOut.length;
  const xp = startingXi.reduce((a, b) => a + b.player!.xpNext!, 0);
  const xpChange = xp - existingStartingXi.reduce((a, b) => a + b.player!.xpNext!, 0);
  const squadValue = (startingXi.reduce((a, b) => a + b.player!.cost!, 0) + bench.reduce((a, b) => a + b.player!.cost!, 0)) / 10;
  const squadValueChange = squadValue - (existingStartingXi.reduce((a, b) => a + b.player!.cost!, 0) + existingBench.reduce((a, b) => a + b.player!.cost!, 0)) / 10;

  return <div className="">
    <div className="grid grid-cols-1 gap-4 mb-4">
      <div className="bg-white border border-gray-300 shadow-lg rounded-lg py-4 px-6 ">
        <table>
          <tbody>
            <tr>
              <td colSpan={5} className="text-center font-mono text-sm text-gray-400 p-2">OUT</td>
              <td colSpan={5} className="text-center font-mono text-sm text-gray-400 p-2">IN</td>
            </tr>
            {Array.from({ length: transferCount }).map((_, i) => {
              const playerIn = playersIn.sort((a, b) => a.player!.position! - b.player!.position!)[i].player!;
              const playerOut = playersOut.sort((a, b) => a.player!.position! - b.player!.position!)[i].player!;
              return <tr key={i} className={playerBg(playerOut)}>
                <td><ChevronRightIcon className="text-red-500 h-6 w-6 ml-2" /></td>
                <td className="font-medium px-2">{playerOut.name}</td>
                <td className="px-2 text-sm">{lookupTeam(playerOut!.team!, allData!.teams.output!).shortName}</td>
                <td className="px-2 text-gray-500 font-mono text-sm text-right">
                  £{(playerOut.cost! / 10).toFixed(1)}
                </td>
                <td className="px-2 text-blue-500 font-mono text-sm text-right">
                  {playerOut.xpNext!.toFixed(1)}
                </td>
                <td><ChevronRightIcon className="text-green-500 h-6 w-6 ml-2" /></td>
                <td className="font-medium px-2">{playerIn.name}</td>
                <td className="px-2 text-sm">{lookupTeam(playerIn!.team!, allData!.teams.output!).shortName}</td>
                <td className="px-2 text-gray-500 font-mono text-sm text-right">
                  £{(playerIn.cost! / 10).toFixed(1)}
                </td>
                <td className="px-2 text-blue-500 font-mono text-sm text-right">
                  {playerIn.xpNext!.toFixed(1)}
                </td>
              </tr>
            })}
          </tbody>
        </table>
      </div>
    </div>

    <div className="bg-white border border-gray-300 shadow-lg rounded-lg py-4 px-6 ">
      <div className="flex justify-between">
        <span>Squad value:</span>
        <div>
          <span className="text-gray-500 font-mono text-sm text-right">£{squadValue.toFixed(1)}m </span>
          {squadValueChange > 0
            ? <span className="text-gray-500 font-mono text-sm text-right">(+£{+squadValueChange.toFixed(1)}m)</span>
            : <span className="text-gray-500 font-mono text-sm text-right">(-£{(squadValueChange * -1).toFixed(1)}m)</span>}
        </div>
      </div>
      <div className="flex justify-between mt-2">
        <span>Bank:</span>
        <div>
          <span className="text-gray-500 font-mono text-sm text-right">£{(bank / 10).toFixed(1)}m </span>
        </div>
      </div>
      <div className="flex justify-between mt-2">
        <span>XP:</span>
        <div className="text-blue-500">
          <span className="font-mono text-sm text-right">{xp.toFixed(1)} </span>
          ({xpChange > 0
            ? <span className="text-green-500 font-mono text-sm text-right">+{xpChange.toFixed(1)}</span>
            : <span className="text-red-500 font-mono text-sm text-right">{xpChange.toFixed(1)}</span>})
        </div>
      </div>
      <div className="flex justify-between mt-2">
        <span>Final XP {transferCost > 0 && <span className="text-xs italic">(-{transferCost} pts)</span>}:</span>
        <span className="font-semibold text-blue-500 font-mono text-sm">{(xp - transferCost).toFixed(1)}</span>
      </div>
    </div>
  </div>;
}

export default ShowTransfers;