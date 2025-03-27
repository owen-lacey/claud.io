namespace FplTeamPicker.Domain.Models;

public record MyTeam
{
    public int FreeTransfers { get; set; }

    public int Bank { get; set; }

    public SelectedTeam SelectedTeam { get; set; } = null!;

    public int Budget => Bank + SelectedTeam.SquadCost;
}