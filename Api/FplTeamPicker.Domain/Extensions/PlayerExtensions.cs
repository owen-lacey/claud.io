using FplTeamPicker.Domain.Models;

namespace FplTeamPicker.Domain.Extensions;

public static class PlayerExtensions
{
    public static void PopulateCostsFrom(this List<Player> players, Team team)
    {
        foreach (var currentPlayer in team.StartingXi)
        {
            var matchedPlayer = players.Single(p => p.Id == currentPlayer.Player.Id);
            matchedPlayer.Cost = currentPlayer.SellingPrice;
        }
        foreach (var currentPlayer in team.Bench)
        {
            var matchedPlayer = players.Single(p => p.Id == currentPlayer.Player.Id);
            matchedPlayer.Cost = currentPlayer.SellingPrice;
        }
    }
}