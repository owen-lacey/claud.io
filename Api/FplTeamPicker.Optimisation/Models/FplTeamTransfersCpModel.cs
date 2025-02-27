using Google.OrTools.Sat;

namespace FplTeamPicker.Optimisation.Models;

public class FplTeamTransfersCpModel : CpModel
{
    /// <summary>
    /// A dictionary of TeamId to a list of variables representing player selections.
    /// This is used to ensure we don't pick too many players from a single team.
    /// </summary>
    public Dictionary<string, List<IntVar>> TeamSelectionCounts { get; set; }= new();

    /// <summary>
    /// A list of player selection vars for the current team, as we might choose to transfer them out.
    /// </summary>
    public List<FplPlayerSelectionVar> ExistingSelections => ExistingSelectedGks
        .Union(ExistingSelectedDefs)
        .Union(ExistingSelectedMids)
        .Union(ExistingSelectedFwds)
        .ToList();

    public List<FplPlayerSelectionVar> TransferSelections => TransferSelectedGks
        .Union(TransferSelectedDefs)
        .Union(TransferSelectedMids)
        .Union(TransferSelectedFwds)
        .ToList();

    public List<FplPlayerSelectionVar> SelectedGks => TransferSelectedGks.Union(ExistingSelectedGks).ToList();
    public List<FplPlayerSelectionVar> SelectedDefs => TransferSelectedDefs.Union(ExistingSelectedDefs).ToList();
    public List<FplPlayerSelectionVar> SelectedMids => TransferSelectedMids.Union(ExistingSelectedMids).ToList();
    public List<FplPlayerSelectionVar> SelectedFwds => TransferSelectedFwds.Union(ExistingSelectedFwds).ToList();

    public List<FplPlayerSelectionVar> Selections => ExistingSelections.Union(TransferSelections).ToList();

    public List<FplPlayerSelectionVar> TransferSelectedGks { get; set; } = new();
    public List<FplPlayerSelectionVar> TransferSelectedDefs { get; set; } = new();
    public List<FplPlayerSelectionVar> TransferSelectedMids { get; set; } = new();
    public List<FplPlayerSelectionVar> TransferSelectedFwds { get; set; } = new();

    public List<FplPlayerSelectionVar> ExistingSelectedGks { get; set; } = new();
    public List<FplPlayerSelectionVar> ExistingSelectedDefs { get; set; } = new();
    public List<FplPlayerSelectionVar> ExistingSelectedMids { get; set; } = new();
    public List<FplPlayerSelectionVar> ExistingSelectedFwds { get; set; } = new();
}