using FplTeamPicker.Domain.Models;

namespace FplTeamPicker.Services.Optimisation.Models;

public class PickFplTeamModel
{
    public PickFplTeamModel(List<Player> players, FplOptions options, int budget)
    {
        Players = players;
        Options = options;
        Budget = budget;
    }
    public List<Player> Players { get; }

    public FplOptions Options { get; }

    public int Budget { get; set; }
}