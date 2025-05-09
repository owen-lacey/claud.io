using FplTeamPicker.Domain.Models;
using FplTeamPicker.Services.Optimisation.Models;

namespace FplTeamPicker.Services.Optimisation.UseCases.Wildcard;

public class WildcardModelInput
{
    public WildcardModelInput(List<Player> players, FplOptions options, int budget)
    {
        Players = players;
        Options = options;
        Budget = budget;
    }
    public List<Player> Players { get; }

    public FplOptions Options { get; }

    public int Budget { get; set; }
}