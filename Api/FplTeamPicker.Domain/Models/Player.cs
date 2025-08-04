namespace FplTeamPicker.Domain.Models;

public record Player
{
    public int Id { get; set; }

    public Position Position { get; set; }

    public int Cost { get; set; }

    public int? ChanceOfPlayingNextRound { get; set; }

    public string FirstName { get; set; } = null!;

    public string SecondName { get; set; } = null!;

    public decimal? Xp { get; set; }

    public decimal SelectedByPercent { get; set; }

    public int Team { get; set; }

    public int SeasonPoints { get; init; }
    
    public int YellowCards { get; set; }
    
    public DateOnly? BirthDate { get; set; }
    
    public int RedCards { get; set; }
    
    public int TransfersOut { get; set; }

    public string Name => $"{FirstName} {SecondName}";

    public bool IsAvailable => ChanceOfPlayingNextRound == null || ChanceOfPlayingNextRound == 100;

    public Dictionary<int, decimal> Predictions { get; set; } = new();
}