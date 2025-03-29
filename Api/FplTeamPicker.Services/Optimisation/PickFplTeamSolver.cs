using FplTeamPicker.Domain.Models;
using FplTeamPicker.Services.Optimisation.Exceptions;
using FplTeamPicker.Services.Optimisation.Models;
using Google.OrTools.Sat;
using LinearExpr = Google.OrTools.Sat.LinearExpr;
namespace FplTeamPicker.Services.Optimisation;

public class PickFplTeamSolver
{
    private readonly PickFplTeamModel _request;

    public PickFplTeamSolver(PickFplTeamModel request)
    {
        _request = request;
    }

    public FplTeamSelection Solve()
    {
        var model = new PickFplTeamCpModel();

        InitialiseVars(model);

        AddConstraints(model);

        AddObjective(model);

        return Solve(model);
    }

    private FplTeamSelection Solve(PickFplTeamCpModel model)
    {
        var cpSolver = new CpSolver();
        var status = cpSolver.Solve(model);
        if (status != CpSolverStatus.Optimal)
        {
            throw OptimisationException.SubOptimalCpSolution(status);
        }

        var output = new FplTeamSelection();

        foreach (var playerSelectionVar in model.Selections)
        {
            if (!cpSolver.BooleanValue((BoolVar)playerSelectionVar.SquadSelected))
            {
                continue;
            }
            var player = _request.Players.Single(p => p.Id == playerSelectionVar.Id);
            var selectedPlayer = new SelectedPlayer
            {
                Player = player,
                SellingPrice = player.Cost
            };

            var teamSelected = cpSolver.BooleanValue((BoolVar)playerSelectionVar.TeamSelected);
            if (teamSelected)
            {
                output.StartingXi.Add(selectedPlayer);
            }
            else
            {
                output.Bench.Add(selectedPlayer);
            }
        }

        var bestPlayers = output.StartingXi.OrderByDescending(p => p.Player.XpNext).Take(2).ToList();
        bestPlayers[0].IsCaptain = true;
        bestPlayers[1].IsViceCaptain = true;

        return output;
    }

    private void InitialiseVars(PickFplTeamCpModel model)
    {
        foreach (var player in _request.Players)
        {
            var squadSelection = GetSelectionVar(model, player);

            if (model.TeamSelectionCounts.ContainsKey(player.Team))
            {
                model.TeamSelectionCounts[player.Team].Add(squadSelection.SquadSelected);
            }
            else
            {
                model.TeamSelectionCounts.Add(player.Team, new List<IntVar> {squadSelection.SquadSelected});
            }
        }
    }

    private void AddObjective(PickFplTeamCpModel model)
    {
        // Get the scale product of the selection booleans and multiply by each player's XI selection, maximizing the total.
        var allPlayerPredictedPoints =
            _request.Players.Select(p => (int)Math.Round(p.XpNext * 100)).ToList();
        var teamPredictedPoints = LinearExpr.WeightedSum(model.Selections.Select(s => s.TeamSelected), allPlayerPredictedPoints);
        model.Maximize(teamPredictedPoints);
    }

    private void AddConstraints(PickFplTeamCpModel model)
    {
        AddPositionConstraints(model);

        AddMaxPerTeamConstraint(model);

        AddCostConstraint(model);

        AddStartingTeamConstraints(model);
    }

    private void AddStartingTeamConstraints(PickFplTeamCpModel model)
    {
        var teamGoalkeepers = LinearExpr.Sum(model.SelectedGks.Select(p => p.TeamSelected));
        var teamDefenders = LinearExpr.Sum(model.SelectedDefs.Select(p => p.TeamSelected));
        var teamMidfielders = LinearExpr.Sum(model.SelectedMids.Select(p => p.TeamSelected));
        var teamForwards = LinearExpr.Sum(model.SelectedFwds.Select(p => p.TeamSelected));
        model.Add(teamGoalkeepers == 1);
        model.Add(teamDefenders >= _request.Options.MinTeamDefenders);
        model.Add(teamMidfielders >= _request.Options.MinTeamMidfielders);
        model.Add(teamForwards >= _request.Options.MinTeamForwards);

        model.Add(
            teamGoalkeepers + teamDefenders + teamMidfielders + teamForwards == _request.Options.UsefulPlayers);

        foreach (var playerSelectionVar in model.Selections)
        {
            model.Add(playerSelectionVar.SquadSelected >= playerSelectionVar.TeamSelected);
        }
    }

    private void AddPositionConstraints(PickFplTeamCpModel model)
    {
        // Sum up the selections for each position, and ensure it matches the number of players required per position.
        model.Add(LinearExpr.Sum(model.SelectedGks.Select(p => p.SquadSelected)) == _request.Options.SquadGoalkeeperCount);
        model.Add(LinearExpr.Sum(model.SelectedDefs.Select(p => p.SquadSelected)) == _request.Options.SquadDefenderCount);
        model.Add(LinearExpr.Sum(model.SelectedMids.Select(p => p.SquadSelected)) == _request.Options.SquadMidfielderCount);
        model.Add(LinearExpr.Sum(model.SelectedFwds.Select(p => p.SquadSelected)) == _request.Options.SquadForwardCount);
    }

    private void AddMaxPerTeamConstraint(PickFplTeamCpModel model)
    {
        // Sum up the selections for each team, and ensure it doesn't exceed the maximum number of players per team.
        foreach (var (_, selections) in model.TeamSelectionCounts)
        {
            model.Add(LinearExpr.Sum(selections) <= _request.Options.MaxPlayersPerTeam);
        }
    }

    private void AddCostConstraint(PickFplTeamCpModel model)
    {
        // Get the scale product of the selection booleans and multiply by each player's cost.
        var allPlayerCosts = _request.Players.Select(p => p.Cost).ToList();
        var allPlayerSelections = LinearExpr.WeightedSum(model.Selections.Select(p => p.SquadSelected), allPlayerCosts);
        model.Add(allPlayerSelections <= _request.Budget);
    }

    private static FplPlayerSelectionVar GetSelectionVar(PickFplTeamCpModel model, Player player)
    {
        var selection = new FplPlayerSelectionVar(model, player);

        switch (player.Position)
        {
            case Position.Goalkeeper:
                model.SelectedGks.Add(selection);
                break;
            case Position.Defender:
                model.SelectedDefs.Add(selection);
                break;
            case Position.Midfielder:
                model.SelectedMids.Add(selection);
                break;
            case Position.Forward:
                model.SelectedFwds.Add(selection);
                break;
        }

        return selection;
    }
}