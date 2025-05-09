using FplTeamPicker.Domain.Models;

namespace FplTeamPicker.Services.Optimisation.UseCases.Tots;

public record TotsModelOutput
{
    public List<SelectedPlayer> StartingXi { get; set; } = [];
}