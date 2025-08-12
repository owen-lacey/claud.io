"use client";

import React, { useContext, useMemo } from 'react';
import { DataContext } from '@/lib/contexts';
import { Fixture, Team } from '@/helpers/api';

const getDifficultyColor = (difficulty: number) => {
  switch (difficulty) {
    case 1: return "bg-green-500";
    case 2: return "bg-green-400";
    case 3: return "bg-yellow-400";
    case 4: return "bg-orange-400";
    case 5: return "bg-red-500";
    default: return "bg-gray-400";
  }
};

const getDifficultyTextColor = (difficulty: number) => {
  switch (difficulty) {
    case 1: return "text-green-500";
    case 2: return "text-green-400";
    case 3: return "text-yellow-400";
    case 4: return "text-orange-400";
    case 5: return "text-red-500";
    default: return "text-gray-400";
  }
};

export default function FixturesWidget() {
  const data = useContext(DataContext);

  const processedData = useMemo(() => {
    if (!data?.teams.isSuccess || !data.teams.output || !data?.fixtures.isSuccess || !data.fixtures.output) {
      return null;
    }

    const teams = data.teams.output;
    const fixtures = data.fixtures.output;
    
    // Create a map of team ID to team for quick lookup
    const teamMap = teams.reduce((acc, team) => {
      if (team.id) acc[team.id] = team;
      return acc;
    }, {} as Record<number, Team>);

    // Group fixtures by team and get next 6 for each team
    const teamFixtures = teams.map(team => {
      if (!team.id) return null;
      
      // Get all fixtures for this team (both home and away)
      const teamFixtureList = fixtures
        .filter(fixture => 
          (fixture.teamHome === team.id || fixture.teamAway === team.id) && 
          !fixture.finished &&
          fixture.gameweek
        )
        .sort((a, b) => (a.gameweek! - b.gameweek!))
        .slice(0, 6)
        .map(fixture => {
          const isHome = fixture.teamHome === team.id;
          const opponentId = isHome ? fixture.teamAway! : fixture.teamHome!;
          const opponent = teamMap[opponentId];
          const difficulty = isHome ? fixture.teamHomeDifficulty! : fixture.teamAwayDifficulty!;
          
          return {
            gameweek: fixture.gameweek!,
            opponent: opponent?.shortName || 'TBD',
            isHome,
            difficulty,
            kickoffTime: fixture.kickoffTime
          };
        });

      return {
        team,
        fixtures: teamFixtureList
      };
    }).filter(Boolean);

    return teamFixtures;
  }, [data]);

  if (!processedData) {
    return (
      <div className="border border-border/50 rounded-xl p-4 bg-card">
        <h2 className="text-lg font-semibold text-foreground mb-4">Team Fixtures</h2>
        <p className="text-muted-foreground">Loading fixtures data...</p>
      </div>
    );
  }

  return (
    <div className="border border-border/50 rounded-xl p-4 bg-card">
      <h2 className="text-lg font-semibold text-foreground mb-4">Next 6 Fixtures</h2>
      
      <div className="space-y-3 max-h-[70vh] overflow-y-auto">
        {processedData.slice(0, 20).map((teamData) => {
          if (!teamData) return null;
          
          return (
            <div key={teamData.team.id} className="border border-border/30 rounded-lg p-3 bg-background/50">
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-medium text-foreground">{teamData.team.name}</h3>
                <span className="text-xs text-muted-foreground">({teamData.team.shortName})</span>
              </div>
              
              <div className="grid grid-cols-6 gap-1">
                {teamData.fixtures.map((fixture, index) => (
                  <div key={index} className="text-center">
                    <div 
                      className={`text-xs py-1 px-1 rounded text-white font-medium ${getDifficultyColor(fixture.difficulty)}`}
                      title={`GW${fixture.gameweek}: ${fixture.isHome ? 'vs' : '@'} ${fixture.opponent} (Difficulty: ${fixture.difficulty}/5)`}
                    >
                      {fixture.isHome ? 'vs' : '@'}
                    </div>
                    <div className="text-xs mt-1 font-mono text-foreground">
                      {fixture.opponent}
                    </div>
                    <div className={`text-xs ${getDifficultyTextColor(fixture.difficulty)} font-semibold`}>
                      GW{fixture.gameweek}
                    </div>
                  </div>
                ))}
                {/* Fill empty slots if less than 6 fixtures */}
                {Array.from({ length: 6 - teamData.fixtures.length }).map((_, index) => (
                  <div key={`empty-${index}`} className="text-center">
                    <div className="text-xs py-1 px-1 rounded bg-gray-600 text-gray-400">
                      -
                    </div>
                    <div className="text-xs mt-1 font-mono text-muted-foreground">
                      TBD
                    </div>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>
      
      <div className="mt-4 p-3 bg-muted/20 rounded-lg">
        <h4 className="text-sm font-medium text-foreground mb-2">Difficulty Legend</h4>
        <div className="flex gap-2 text-xs">
          <span className="flex items-center gap-1">
            <div className="w-3 h-3 bg-green-500 rounded"></div>
            Easy (1-2)
          </span>
          <span className="flex items-center gap-1">
            <div className="w-3 h-3 bg-yellow-400 rounded"></div>
            Medium (3)
          </span>
          <span className="flex items-center gap-1">
            <div className="w-3 h-3 bg-orange-400 rounded"></div>
            Hard (4)
          </span>
          <span className="flex items-center gap-1">
            <div className="w-3 h-3 bg-red-500 rounded"></div>
            Very Hard (5)
          </span>
        </div>
      </div>
    </div>
  );
}
