namespace FplTeamPicker.Models;

public class FplTeamSelection
{
    public List<FplPlayer> Squad { get; set; } = new();

    public List<FplPlayer> Team { get; set; } = new();

    public FplPlayer Captain => Squad.MaxBy(t => t.PredictedPoints)!;

    public long Cost => Squad.Sum(sp => sp.Cost);

    public decimal PredictedPoints => Squad.Sum(t => t.PredictedPoints);
}