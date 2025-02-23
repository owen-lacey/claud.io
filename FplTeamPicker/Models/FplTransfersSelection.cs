namespace FplTeamPicker.Models;

public class FplTransfersSelection
{
    public List<FplPlayer> PlayersKept { get; set; } = new();

    public List<FplPlayer> PlayersOut { get; set; } = new();

    public List<FplPlayer> PlayersIn { get; set; } = new();

    public List<FplPlayer> Squad => PlayersKept.Union(PlayersIn).ToList();

    public List<FplPlayer> Team { get; set; } = new();

    public FplPlayer Captain => Team.MaxBy(t => t.PredictedPoints)!;

    public List<FplPlayer> Bench => Squad.Except(Team).ToList();

    public long Cost => Squad.Sum(sp => sp.Cost);

    public decimal PredictedPoints => Team.Sum(t => t.PredictedPoints);
}