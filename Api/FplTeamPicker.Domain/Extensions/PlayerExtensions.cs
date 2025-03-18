using FplTeamPicker.Domain.Models;

namespace FplTeamPicker.Domain.Extensions;

public static class PlayerExtensions
{
    public static void PopulateCostsFrom(this List<Player> players, SelectedTeam selectedTeam)
    {
        foreach (var currentPlayer in selectedTeam.StartingXi)
        {
            var matchedPlayer = players.Single(p => p.Id == currentPlayer.Player.Id);
            matchedPlayer.Cost = currentPlayer.SellingPrice;
        }
        foreach (var currentPlayer in selectedTeam.Bench)
        {
            var matchedPlayer = players.Single(p => p.Id == currentPlayer.Player.Id);
            matchedPlayer.Cost = currentPlayer.SellingPrice;
        }
    }
}