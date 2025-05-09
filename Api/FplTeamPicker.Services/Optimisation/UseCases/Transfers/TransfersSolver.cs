using FplTeamPicker.Domain.Models;
using FplTeamPicker.Services.Optimisation.Exceptions;
using FplTeamPicker.Services.Optimisation.Models;
using Google.OrTools.Sat;

namespace FplTeamPicker.Services.Optimisation.UseCases.Transfers;

public class TransfersSolver
{
    private readonly TransfersModelInput _input;

    public TransfersSolver(TransfersModelInput input)
    {
        _input = input;
    }

    public TransfersModelOutput Solve()
    {
        var model = new TransfersCpModel();

        InitialiseVars(model);

        AddConstraints(model);

        AddObjective(model);

        return Solve(model);
    }

    private TransfersModelOutput Solve(TransfersCpModel model)
    {
        var cpSolver = new CpSolver();
        var status = cpSolver.Solve(model);
        if (status != CpSolverStatus.Optimal)
        {
            throw OptimisationException.SubOptimalCpSolution(status);
        }

        var output = new TransfersModelOutput();
        foreach (var playerSelectionVar in model.Selections)
        {
            if (!cpSolver.BooleanValue((BoolVar)playerSelectionVar.SquadSelected))
            {
                continue;
            }
            var player = _input.AllPlayers.Single(p => p.Id == playerSelectionVar.Id);
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

        return output;
    }

    private void InitialiseVars(TransfersCpModel model)
    {
        foreach (var existingPlayer in _input.ExistingPlayers)
        {
            var squadSelection = GetSelectionVar(model, existingPlayer, isExisting: true);

            InitTeamSelectionCount(model, existingPlayer, squadSelection.SquadSelected);
        }

        foreach (var otherPlayer in _input.OtherPlayers)
        {
            var squadSelection = GetSelectionVar(model, otherPlayer, isExisting: false);

            InitTeamSelectionCount(model, otherPlayer, squadSelection.SquadSelected);
        }
    }

    private static void InitTeamSelectionCount(TransfersCpModel model, Player existingPlayer,
        IntVar squadSelected)
    {
        if (model.TeamSelectionCounts.ContainsKey(existingPlayer.Team))
        {
            model.TeamSelectionCounts[existingPlayer.Team].Add(squadSelected);
        }
        else
        {
            model.TeamSelectionCounts.Add(existingPlayer.Team, [squadSelected]);
        }
    }

    private void AddObjective(TransfersCpModel model)
    {
        // Get the scale product of the selection booleans and multiply by each player's XI selection, maximizing the total.
        const int multiplier = 100;
        var allPlayerPredictedPoints = _input.AllPlayers.Select(p => (int)Math.Round(p.XpNext * multiplier)).ToList();
        var predictedPointsForTeam = LinearExpr.WeightedSum(model.Selections.Select(p => p.TeamSelected), allPlayerPredictedPoints);

        // (# transfers - free transfers) * op
        var transferSum = LinearExpr.Sum(model.TransferSelections.Select(p => p.SquadSelected));

        var penaliseTransfers = model.NewBoolVar("penaliseTransfers");
        model.Add(transferSum > _input.NumberTransfers).OnlyEnforceIf(penaliseTransfers);
        model.Add(transferSum <= _input.NumberTransfers).OnlyEnforceIf(penaliseTransfers.Not());
        var penalisedTransferCount = model.NewIntVar(0, _input.Options.SquadCount, "penalisedTransferCount");
        model.Add(penalisedTransferCount == transferSum - _input.NumberTransfers).OnlyEnforceIf(penaliseTransfers);
        var transferPenalty = penalisedTransferCount * _input.Options.TransferPointsPenalty * multiplier;
        
        var allPlayerCosts = _input.AllPlayers.Select(p => p.Cost).ToList();
        var teamCost = LinearExpr.WeightedSum(model.Selections.Select(p => p.SquadSelected), allPlayerCosts);
        var budgetRemaining = _input.Budget - teamCost;
        model.Maximize(predictedPointsForTeam - transferPenalty + budgetRemaining);
    }

    private void AddConstraints(TransfersCpModel model)
    {
        AddPositionConstraints(model);

        AddMaxPerTeamConstraint(model);

        AddCostConstraint(model);

        AddStartingTeamConstraints(model);
    }

    private void AddStartingTeamConstraints(TransfersCpModel model)
    {
        var teamGoalkeepers = LinearExpr.Sum(model.SelectedGks.Select(p => p.TeamSelected));
        var teamDefenders = LinearExpr.Sum(model.SelectedDefs.Select(p => p.TeamSelected));
        var teamMidfielders = LinearExpr.Sum(model.SelectedMids.Select(p => p.TeamSelected));
        var teamForwards = LinearExpr.Sum(model.SelectedFwds.Select(p => p.TeamSelected));
        model.Add(teamGoalkeepers == 1);
        model.Add(teamDefenders >= _input.Options.MinTeamDefenders);
        model.Add(teamMidfielders >= _input.Options.MinTeamMidfielders);
        model.Add(teamForwards >= _input.Options.MinTeamForwards);

        model.Add(
            teamGoalkeepers + teamDefenders + teamMidfielders + teamForwards == _input.Options.UsefulPlayers);

        foreach (var playerSelectionVar in model.Selections)
        {
            model.Add(playerSelectionVar.SquadSelected >= playerSelectionVar.TeamSelected);
        }
    }

    private void AddPositionConstraints(TransfersCpModel model)
    {
        // Sum up the selections for each position, and ensure it matches the number of players required per position.
        model.Add(LinearExpr.Sum(model.SelectedGks.Select(p => p.SquadSelected)) == _input.Options.SquadGoalkeeperCount);
        model.Add(LinearExpr.Sum(model.SelectedDefs.Select(p => p.SquadSelected)) == _input.Options.SquadDefenderCount);
        model.Add(LinearExpr.Sum(model.SelectedMids.Select(p => p.SquadSelected)) == _input.Options.SquadMidfielderCount);
        model.Add(LinearExpr.Sum(model.SelectedFwds.Select(p => p.SquadSelected)) == _input.Options.SquadForwardCount);
    }

    private void AddMaxPerTeamConstraint(TransfersCpModel model)
    {
        // Sum up the selections for each team, and ensure it doesn't exceed the maximum number of players per team.
        foreach (var (_, selections) in model.TeamSelectionCounts)
        {
            model.Add(LinearExpr.Sum(selections) <= _input.Options.MaxPlayersPerTeam);
        }
    }

    private void AddCostConstraint(TransfersCpModel model)
    {
        // Get the scale product of the selection booleans and multiply by each player's cost.
        var allPlayerCosts = _input.AllPlayers.Select(p => p.Cost).ToList();
        var allPlayerCost = LinearExpr.WeightedSum(model.Selections.Select(s => s.SquadSelected), allPlayerCosts);

        model.Add(allPlayerCost <= _input.Budget);
    }

    private SquadSelectionVar GetSelectionVar(TransfersCpModel model, Player player, bool isExisting)
    {
        var selection = new SquadSelectionVar(model, player);

        List<SquadSelectionVar> listOfPlayersToAddTo = player.Position switch
        {
            Position.Goalkeeper => isExisting ? model.ExistingSelectedGks : model.TransferSelectedGks,
            Position.Defender => isExisting ? model.ExistingSelectedDefs : model.TransferSelectedDefs,
            Position.Midfielder => isExisting ? model.ExistingSelectedMids : model.TransferSelectedMids,
            Position.Forward => isExisting ? model.ExistingSelectedFwds : model.TransferSelectedFwds,
            _ => throw new ArgumentOutOfRangeException()
        };

        if (_input.Options.GetRidOf != null && player.Name == _input.Options.GetRidOf.Name)
        {
            model.Add(selection.SquadSelected == 0);
        }
        listOfPlayersToAddTo.Add(selection);

        return selection;
    }
}