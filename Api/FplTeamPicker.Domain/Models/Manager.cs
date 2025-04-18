namespace FplTeamPicker.Domain.Models;

public record Manager()
{
    public int Id { get; set; }

    public int Cost { get; set; }

    public string FirstName { get; set; } = null!;

    public string SecondName { get; set; } = null!;

    public decimal XpNext { get; set; }

    public decimal XpThis { get; set; }

    public int Team { get; set; }

    public string Name => $"{FirstName} {SecondName}";
}