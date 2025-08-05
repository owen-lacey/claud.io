"use client";

import { useContext, useState } from "react";
import { playerBg } from "../helpers/styles";
import { lookupTeam } from "../helpers/lookups";
import RivalSelectionCount from "./players/RivalSelectionCount";
import { LoadingCard } from "./utils/Loading";
import { DataContext, RivalTeamsContext } from "@/lib/contexts";

function MyTeam() {
  const allData = useContext(DataContext);
  const rivalTeams = useContext(RivalTeamsContext);

  if (!allData?.myTeam.output || !allData?.teams.output) {
    return <LoadingCard />;
  }
  const { myTeam, teams } = allData;

  const showRivalSelectionCount = rivalTeams.length > 0;

  return <div className="bg-card border border-border shadow-lg rounded-lg p-4 flex flex-col">
    <div className="flex gap-1 justify-between mb-2">
      <div className="w-19 rounded-md bg-linear-to-br from-cyan-600 to-blue-900 p-2 text-gray-100 font-mono flex flex-col items-center justify-center">
        <div className="text-lg">£{(myTeam.output!.bank! / 10).toFixed(1)}</div>
        <div className="text-xs uppercase">bank</div>
      </div>
      <div className="w-19 rounded-md bg-linear-to-br from-cyan-600 to-blue-900 p-2 text-gray-100 font-mono flex flex-col items-center justify-center">
        <div className="text-lg">£{(myTeam.output!.budget! / 10).toFixed(1)}</div>
        <div className="text-xs uppercase">value</div>
      </div>
      <div className="w-19 rounded-md bg-linear-to-br from-cyan-600 to-blue-900 p-2 text-gray-100 font-mono flex flex-col items-center justify-center">
        <div className="text-lg">{myTeam.output!.freeTransfers}</div>
        <div className="text-xs uppercase">ft</div>
      </div>
    </div>
    <table>
      <tbody>
        <tr className="font-mono text-sm text-gray-400 uppercase text-center"><td className="p-2" key={0} colSpan={99}>XI</td></tr>
        {myTeam.output!.selectedSquad!.startingXi!.map((player, index) => (
          <tr key={index} className={playerBg(player.player!)}>
            <td className="font-medium flex justify-between items-center px-2">
              <div>{player.player!.name}</div>
              {showRivalSelectionCount ? <RivalSelectionCount rivalTeams={rivalTeams} playerId={player.player!.id!} /> : <></>}
            </td>
            <td className="text-sm">{lookupTeam(player.player!.team!, teams.output!).shortName}</td>
            <td className="text-gray-500 font-mono text-sm text-right">£{(player.player!.cost! / 10).toFixed(1)}</td>
            <td className="text-blue-500 font-mono text-sm text-right">{player.player!.xp?.toFixed(1)}</td>
          </tr>
        ))}
        <tr className="font-mono text-sm text-gray-400 uppercase text-center"><td className="p-2" key={0} colSpan={99}>bench</td></tr>
        {myTeam.output!.selectedSquad!.bench!.map((player, index) => (
          <tr key={index} className={playerBg(player.player!)}>
            <td className="font-medium flex justify-between items-center px-2">
              <div>{player.player!.name}</div>
              {showRivalSelectionCount ? <RivalSelectionCount rivalTeams={rivalTeams} playerId={player.player!.id!} /> : <></>}
            </td>
            <td className="text-sm">{lookupTeam(player.player!.team!, teams.output!).shortName}</td>
            <td className="text-gray-500 font-mono text-sm text-right">£{(player.player!.cost! / 10).toFixed(1)}</td>
            <td className="text-blue-500 font-mono text-sm text-right">{player.player!.xp?.toFixed(1)}</td>
          </tr>
        ))}
      </tbody>
    </table>
    <div className="flex justify-end py-4 text-blue-500 text-sm">
      <div>XP (bench boost):&nbsp;</div>
      <div className="font-mono">{myTeam.output!.selectedSquad!.predictedPoints?.toFixed(1)} ({myTeam.output!.selectedSquad!.benchBoostPredictedPoints?.toFixed(1)})</div>
    </div>
  </div>
}

export default MyTeam;