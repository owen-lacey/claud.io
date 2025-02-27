namespace FplTeamPicker.Domain.Models;

public record Team
{
    public ICollection<Player> Players { get; set; }
}