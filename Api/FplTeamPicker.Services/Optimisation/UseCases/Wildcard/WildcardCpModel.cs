using FplTeamPicker.Services.Optimisation.Models;
using Google.OrTools.Sat;

namespace FplTeamPicker.Services.Optimisation.UseCases.Wildcard;

public class WildcardCpModel : CpModel
{
    public List<SquadSelectionVar> SelectedGks { get; set; } = [];
    public List<SquadSelectionVar> SelectedDefs { get; set; } = [];
    public List<SquadSelectionVar> SelectedMids { get; set; } = [];
    public List<SquadSelectionVar> SelectedFwds { get; set; } = [];

    public List<SquadSelectionVar> Selections => SelectedGks
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