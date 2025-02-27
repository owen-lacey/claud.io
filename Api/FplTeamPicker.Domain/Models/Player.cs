namespace FplTeamPicker.Domain.Models;

public record Player
{
    public bool IsViceCaptain { get; set; }

    public bool IsCaptain { get; set; }

    public int Id { get; set; }

    public int Position { get; set; }

    /// <summary>
    /// How much the player was bought for
    /// </summary>
    public int InitialPurchasePrice { get; set; }

    /// <summary>
    /// How much we can sell the player for
    /// </summary>
    public int SellingPrice { get; set; }

    /// <summary>
    /// How much the player costs now
    /// </summary>
    public int PurchasePrice { get; set; }

    public string FirstName { get; set; } = null!;

    public string SecondName { get; set; } = null!;

    public string XpNext { get; set; }

    public string XpThis { get; set; }
}