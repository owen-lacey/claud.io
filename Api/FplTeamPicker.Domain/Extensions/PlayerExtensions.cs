using FplTeamPicker.Domain.Models;

namespace FplTeamPicker.Domain.Extensions;

public static class PlayerExtensions
{
    public static void PopulateCostsFrom(this List<Player> players, MyTeam myTeam)
    {
        foreach (var currentPlayer in myTeam.SelectedTeam.StartingXi)
        {
            var matchedPlayer = players.Single(p => p.Id == currentPlayer.Player.Id);
            matchedPlayer.Cost = currentPlayer.SellingPrice;
        }
        foreach (var currentPlayer in myTeam.SelectedTeam.Bench)
        {
            var matchedPlayer = players.Single(p => p.Id == currentPlayer.Player.Id);
            matchedPlayer.Cost = currentPlayer.SellingPrice;
        }
    }
}