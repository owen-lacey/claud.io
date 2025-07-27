"use client";

import { memo, useContext } from "react";
import { playerBg } from "../helpers/styles";
import { lookupTeam } from "../helpers/lookups";
import SelectionIcon from "./players/SelectionIcon";
import RivalSelectionCount from "./players/RivalSelectionCount";
import { LoadingCard } from "./utils/Loading";
import { DataContext, RivalTeamsContext } from "@/lib/contexts";

function Players() {
  const allData = useContext(DataContext);
  const rivalTeams = useContext(RivalTeamsContext);

  if (!allData?.players.output || !allData?.teams.output) {
    return <LoadingCard />;
  }

  const { players, teams, myTeam } = allData;
  const showRivalSelectionCount = rivalTeams.length > 0;

  return <div className="bg-card border border-border shadow-lg rounded-lg p-4 flex flex-col">
    <h2 className="text-lg font-semibold mb-2 text-foreground">Players</h2>
    <table>
      <tbody>
        {players.output!.slice(0, 50).map((player, index) => (
          <tr key={index} className={playerBg(player)}>
            <td className="font-medium flex justify-between items-center px-2">
              <div className="flex items-center px-2">
                {player.name}
                <SelectionIcon player={player} team={myTeam.output?.selectedSquad || null} />
              </div>
              {showRivalSelectionCount ? <RivalSelectionCount rivalTeams={rivalTeams} playerId={player!.id!} /> : <></>}
            </td>
            <td className="text-sm">{lookupTeam(player.team!, teams.output!).shortName}</td>
            <td className="text-gray-500 font-mono text-sm text-right">£{(player.cost! / 10).toFixed(1)}</td>
            <td className="text-blue-500 font-mono text-sm text-right">{player.xpNext!.toFixed(1)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  </div>
}

export default Players;