using FplTeamPicker.Domain.Models;

namespace FplTeamPicker.Services.Optimisation.Models;

public class PickFplTeamModel(List<Player> players, FplOptions options, int budget)
{
    public List<Player> Players { get; } = players;

    public FplOptions Options { get; } = options;

    public int Budget { get; set; } = budget;
}