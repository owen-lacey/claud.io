
using FplTeamPicker.Domain.Models;

namespace FplTeamPicker.Services.Optimisation.Models;

public class FplOptions
{
    public int MaxPlayersPerTeam { get; set; } = int.MaxValue;

    public int SquadGoalkeeperCount { get; set; }

    public int SquadDefenderCount { get; set; }

    public int SquadMidfielderCount { get; set; }

    public int SquadForwardCount { get; set; }

    public int SquadCount => SquadGoalkeeperCount + SquadDefenderCount + SquadMidfielderCount + SquadForwardCount;

    public int MinTeamDefenders { get; set; }

    public int MinTeamMidfielders { get; set; }

    public int MinTeamForwards { get; set; }

    /// <summary>
    /// The number of players that won't be in the starting XI, but can come off the bench.
    /// </summary>
    public int UsefulBenchPlayers { get; set; }

    public int StartingTeamCount { get; set; }

    public int TransferPointsPenalty { get; set; }

    public int UsefulPlayers => StartingTeamCount + UsefulBenchPlayers;

    public static FplOptions RealWorld => new()
    {
        MaxPlayersPerTeam = 3,
        SquadGoalkeeperCount = 2,
        SquadDefenderCount = 5,
        SquadMidfielderCount = 5,
        SquadForwardCount = 3,
        MinTeamDefenders = 3,
        MinTeamMidfielders = 2,
        MinTeamForwards = 1,
        StartingTeamCount = 11,
        TransferPointsPenalty = 4
    };

    public static FplOptions RealWorldXi => new()
    {
        MaxPlayersPerTeam = 3,
        MinTeamDefenders = 3,
        MinTeamMidfielders = 2,
        MinTeamForwards = 1,
        SquadDefenderCount = 5,
        SquadMidfielderCount = 5,
        SquadForwardCount = 3,
        StartingTeamCount = 11
    };

    public Player? GetRidOf { get; set; }
}