using FplTeamPicker.Services.Optimisation.Models;
using Google.OrTools.Sat;

namespace FplTeamPicker.Services.Optimisation.UseCases.Transfers;

public class TransfersCpModel : CpModel
{
    /// <summary>
    /// A dictionary of TeamId to a list of variables representing player selections.
    /// This is used to ensure we don't pick too many players from a single team.
    /// </summary>
    public Dictionary<int, List<IntVar>> TeamSelectionCounts { get; set; }= new();

    /// <summary>
    /// A list of player selection vars for the current team, as we might choose to transfer them out.
    /// </summary>
    public List<SquadSelectionVar> ExistingSelections => ExistingSelectedGks
        .Union(ExistingSelectedDefs)
        .Union(ExistingSelectedMids)
        .Union(ExistingSelectedFwds)
        .ToList();

    public List<SquadSelectionVar> TransferSelections => TransferSelectedGks
        .Union(TransferSelectedDefs)
        .Union(TransferSelectedMids)
        .Union(TransferSelectedFwds)
        .ToList();

    public List<SquadSelectionVar> SelectedGks => TransferSelectedGks.Union(ExistingSelectedGks).ToList();
    public List<SquadSelectionVar> SelectedDefs => TransferSelectedDefs.Union(ExistingSelectedDefs).ToList();
    public List<SquadSelectionVar> SelectedMids => TransferSelectedMids.Union(ExistingSelectedMids).ToList();
    public List<SquadSelectionVar> SelectedFwds => TransferSelectedFwds.Union(ExistingSelectedFwds).ToList();

    public List<SquadSelectionVar> Selections => ExistingSelections.Union(TransferSelections)
        .OrderBy(s => s.Id)
        .ToList();

    public List<SquadSelectionVar> TransferSelectedGks { get; set; } = [];
    public List<SquadSelectionVar> TransferSelectedDefs { get; set; } = [];
    public List<SquadSelectionVar> TransferSelectedMids { get; set; } = [];
    public List<SquadSelectionVar> TransferSelectedFwds { get; set; } = [];

    public List<SquadSelectionVar> ExistingSelectedGks { get; set; } = [];
    public List<SquadSelectionVar> ExistingSelectedDefs { get; set; } = [];
    public List<SquadSelectionVar> ExistingSelectedMids { get; set; } = [];
    public List<SquadSelectionVar> ExistingSelectedFwds { get; set; } = [];
}