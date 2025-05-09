using FplTeamPicker.Domain.Models;
using FplTeamPicker.Services.Optimisation.Exceptions;
using FplTeamPicker.Services.Optimisation.Models;
using Google.OrTools.Sat;

namespace FplTeamPicker.Services.Optimisation.UseCases.Tots;

public class TotsSolver(TotsModelInput input)
{
    private readonly TotsModelInput _input = input;

    public TotsModelOutput Solve()
    {
        var model = new TotsCpModel();

        InitialiseVars(model);

        AddConstraints(model);

        AddObjective(model);

        return Solve(model);
    }

    private TotsModelOutput Solve(TotsCpModel model)
    {
        var cpSolver = new CpSolver();
        var status = cpSolver.Solve(model);
        if (status != CpSolverStatus.Optimal)
        {
            throw OptimisationException.SubOptimalCpSolution(status);
        }

        var output = new TotsModelOutput();

        foreach (var playerSelectionVar in model.Selections)
        {
            if (!cpSolver.BooleanValue((BoolVar)playerSelectionVar.Selected))
            {
                continue;
            }
            var player = _input.Players.Single(p => p.Id == playerSelectionVar.Id);
            var selectedPlayer = new SelectedPlayer
            {
                Player = player,
                SellingPrice = player.Cost
            };
            output.StartingXi.Add(selectedPlayer);
        }

        return output;
    }

    private void InitialiseVars(TotsCpModel model)
    {
        foreach (var player in _input.Players)
        {
            var squadSelection = GetSelectionVar(model, player);

            if (model.TeamSelectionCounts.ContainsKey(player.Team))
            {
                model.TeamSelectionCounts[player.Team].Add(squadSelection.Selected);
            }
            else
            {
                model.TeamSelectionCounts.Add(player.Team, [squadSelection.Selected]);
            }
        }
    }

    private void AddObjective(TotsCpModel model)
    {
        // Get the scale product of the selection booleans and multiply by each player's XI selection, maximizing the total.
        var allPlayerPredictedPoints =
            _input.Players.Select(p => p.SeasonPoints).ToList();
        
        var teamPredictedPoints = LinearExpr.WeightedSum(model.Selections.Select(s => s.Selected), allPlayerPredictedPoints);
        model.Maximize(teamPredictedPoints);
    }

    private void AddConstraints(TotsCpModel model)
    {
        AddMaxPerTeamConstraint(model);

        AddCostConstraint(model);

        AddTeamConstraints(model);
    }

    private void AddTeamConstraints(TotsCpModel model)
    {
        var teamGoalkeepers = LinearExpr.Sum(model.SelectedGks.Select(p => p.Selected));
        var teamDefenders = LinearExpr.Sum(model.SelectedDefs.Select(p => p.Selected));
        var teamMidfielders = LinearExpr.Sum(model.SelectedMids.Select(p => p.Selected));
        var teamForwards = LinearExpr.Sum(model.SelectedFwds.Select(p => p.Selected));
        model.Add(teamGoalkeepers == 1);
        model.Add(teamDefenders >= _input.Options.MinTeamDefenders);
        model.Add(teamMidfielders >= _input.Options.MinTeamMidfielders);
        model.Add(teamForwards >= _input.Options.MinTeamForwards);
        model.Add(teamDefenders <= _input.Options.SquadDefenderCount);
        model.Add(teamMidfielders <= _input.Options.SquadMidfielderCount);
        model.Add(teamForwards <= _input.Options.SquadForwardCount);

        model.Add(
            teamGoalkeepers + teamDefenders + teamMidfielders + teamForwards == _input.Options.UsefulPlayers);
    }

    private void AddMaxPerTeamConstraint(TotsCpModel model)
    {
        // Sum up the selections for each team, and ensure it doesn't exceed the maximum number of players per team.
        foreach (var (_, selections) in model.TeamSelectionCounts)
        {
            model.Add(LinearExpr.Sum(selections) <= _input.Options.MaxPlayersPerTeam);
        }
    }

    private void AddCostConstraint(TotsCpModel model)
    {
        // Get the scale product of the selection booleans and multiply by each player's cost.
        var allPlayerCosts = _input.Players.Select(p => p.Cost).ToList();
        var allPlayerSelections = LinearExpr.WeightedSum(model.Selections.Select(p => p.Selected), allPlayerCosts);
        model.Add(allPlayerSelections <= _input.Budget);
    }

    private static TeamSelectionVar GetSelectionVar(TotsCpModel model, Player player)
    {
        var selection = new TeamSelectionVar(model, player);

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