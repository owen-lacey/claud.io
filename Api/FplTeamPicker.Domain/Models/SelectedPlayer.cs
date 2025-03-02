namespace FplTeamPicker.Domain.Models;

public record SelectedPlayer
{
    public bool IsViceCaptain { get; set; }

    public bool IsCaptain { get; set; }

    public Player Player { get; set; } = null!;

    public int SellingPrice { get; set; }
}