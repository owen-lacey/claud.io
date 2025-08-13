using FplTeamPicker.Domain.Models;
using FplTeamPicker.Services.Optimisation.Models;

namespace FplTeamPicker.Services.Optimisation.UseCases.Wildcard;

public class WildcardModelInput
{
    public WildcardModelInput(List<Player> players, FplOptions options, int budget, List<int> locks)
    {
        Players = players;
        Options = options;
        Budget = budget;
        Locks = locks;
    }
    public List<Player> Players { get; }

    public FplOptions Options { get; }

    public int Budget { get; set; }

    /// <summary>
    /// Gets or sets the list of player IDs that are locked in the team.
    /// </summary>
    public List<int> Locks { get; set; } = [];
}