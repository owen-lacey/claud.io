using FplTeamPicker.Services.Optimisation.Models;
using Google.OrTools.Sat;

namespace FplTeamPicker.Services.Optimisation.UseCases.Tots;

public class TotsCpModel : CpModel
{
    public List<TeamSelectionVar> SelectedGks { get; set; } = [];
    public List<TeamSelectionVar> SelectedDefs { get; set; } = [];
    public List<TeamSelectionVar> SelectedMids { get; set; } = [];
    public List<TeamSelectionVar> SelectedFwds { get; set; } = [];

    public List<TeamSelectionVar> Selections => SelectedGks
        .Union(SelectedDefs)
        .Union(SelectedMids)
        .Union(SelectedFwds)
        .OrderBy(s => s.Id)
        .ToList();

    /// <summary>
    /// A dictionary of TeamId to a list of variables representing player selections.
    /// This is used to ensure we don't pick too many players from a single team.
    /// </summary>
    public Dictionary<int, List<IntVar>> TeamSelectionCounts { get; set; } = new();
}