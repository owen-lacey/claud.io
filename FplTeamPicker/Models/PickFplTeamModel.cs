namespace FplTeamPicker.Models;

public class PickFplTeamModel
{
    public PickFplTeamModel(List<FplPlayer> players, FplOptions options, int budget)
    {
        Players = players;
        Options = options;
        Budget = budget;
    }

    public List<FplPlayer> Players { get; }

    public FplOptions Options { get; }

    public int Budget { get; set; }
}