using FplTeamPicker.Domain.Models;
using FplTeamPicker.Services.Optimisation.Exceptions;
using FplTeamPicker.Services.Optimisation.Models;
using Google.OrTools.LinearSolver;
using Google.OrTools.Sat;
using LinearExpr = Google.OrTools.Sat.LinearExpr;
using SumArray = Google.OrTools.Sat.SumArray;

namespace FplTeamPicker.Services.Optimisation;

public class FplTeamTransfersSolver
{
    private readonly FplTeamTransfersRequest _request;

    public FplTeamTransfersSolver(FplTeamTransfersRequest request)
    {
        _request = request;
    }

    public FplTransfersSelection Solve()
    {
        var model = new FplTeamTransfersCpModel();

        InitialiseVars(model);

        AddConstraints(model);

        AddObjective(model);

        return Solve(model);
    }

    private FplTransfersSelection Solve(FplTeamTransfersCpModel model)
    {
        var cpSolver = new CpSolver();
        var status = cpSolver.Solve(model);
        if (status != CpSolverStatus.Optimal)
        {
            throw OptimisationException.SubOptimalCpSolution(status);
        }

        var output = new FplTransfersSelection();
        foreach (var playerSelectionVar in model.Selections)
        {
            if (!cpSolver.BooleanValue(playerSelectionVar.SquadSelected))
            {
                continue;
            }
            var player = _request.AllPlayers.Single(p => p.Id == playerSelectionVar.Id);
            var selectedPlayer = new SelectedPlayer
            {
                Player = player,
                SellingPrice = player.Cost
            };

            var teamSelected = cpSolver.BooleanValue(playerSelectionVar.TeamSelected);
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

    private void InitialiseVars(FplTeamTransfersCpModel model)
    {
        foreach (var existingPlayer in _request.ExistingPlayers)
        {
            var squadSelection = GetSelectionVar(model, existingPlayer, isExisting: true);

            InitTeamSelectionCount(model, existingPlayer, squadSelection.SquadSelected);
        }

        foreach (var otherPlayer in _request.OtherPlayers)
        {
            var squadSelection = GetSelectionVar(model, otherPlayer, isExisting: false);

            InitTeamSelectionCount(model, otherPlayer, squadSelection.SquadSelected);
        }
    }

    private static void InitTeamSelectionCount(FplTeamTransfersCpModel model, Player existingPlayer,
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

    private void AddObjective(FplTeamTransfersCpModel model)
    {
        // Get the scale product of the selection booleans and multiply by each player's XI selection, maximizing the total.
        var allPlayerPredictedPoints = _request.AllPlayers.Select(p => (int)Math.Round(p.XpNext * 100)).ToList();
        var predictedPointsForTeam = LinearExpr.ScalProd(model.Selections.Select(p => p.TeamSelected), allPlayerPredictedPoints);
        model.Maximize(predictedPointsForTeam);
    }

    private void AddConstraints(FplTeamTransfersCpModel model)
    {
        AddPositionConstraints(model);

        AddMaxPerTeamConstraint(model);

        AddCostConstraint(model);

        AddStartingTeamConstraints(model);

        AddTransferConstraints(model);
    }

    private void AddTransferConstraints(FplTeamTransfersCpModel model)
    {
        var transferredPlayers = new SumArray(model.TransferSelections.Select(s => s.SquadSelected));

        // Can't transfer more players than free transfers allowed.
        model.Add(transferredPlayers <= _request.NumberTransfers);
    }

    private void AddStartingTeamConstraints(FplTeamTransfersCpModel model)
    {
        var teamGoalkeepers = new SumArray(model.SelectedGks.Select(p => p.TeamSelected));
        var teamDefenders = new SumArray(model.SelectedDefs.Select(p => p.TeamSelected));
        var teamMidfielders = new SumArray(model.SelectedMids.Select(p => p.TeamSelected));
        var teamForwards = new SumArray(model.SelectedFwds.Select(p => p.TeamSelected));
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

    private void AddPositionConstraints(FplTeamTransfersCpModel model)
    {
        // Sum up the selections for each position, and ensure it matches the number of players required per position.
        model.Add(new SumArray(model.SelectedGks.Select(p => p.SquadSelected)) == _request.Options.SquadGoalkeeperCount);
        model.Add(new SumArray(model.SelectedDefs.Select(p => p.SquadSelected)) == _request.Options.SquadDefenderCount);
        model.Add(new SumArray(model.SelectedMids.Select(p => p.SquadSelected)) == _request.Options.SquadMidfielderCount);
        model.Add(new SumArray(model.SelectedFwds.Select(p => p.SquadSelected)) == _request.Options.SquadForwardCount);
    }

    private void AddMaxPerTeamConstraint(FplTeamTransfersCpModel model)
    {
        // Sum up the selections for each team, and ensure it doesn't exceed the maximum number of players per team.
        foreach (var (_, selections) in model.TeamSelectionCounts)
        {
            model.Add(new SumArray(selections) <= _request.Options.MaxPlayersPerTeam);
        }
    }

    private void AddCostConstraint(FplTeamTransfersCpModel model)
    {
        // Get the scale product of the selection booleans and multiply by each player's cost.
        var allPlayerCosts = _request.AllPlayers.Select(p => p.Cost).ToList();
        var allPlayerCost = LinearExpr.ScalProd(model.Selections.Select(s => s.SquadSelected), allPlayerCosts);

        model.Add(allPlayerCost <= _request.Budget);
    }

    private FplPlayerSelectionVar GetSelectionVar(FplTeamTransfersCpModel model, Player player, bool isExisting)
    {
        var selection = new FplPlayerSelectionVar(model, player);

        List<FplPlayerSelectionVar> listOfPlayersToAddTo = player.Position switch
        {
            Position.Goalkeeper => isExisting ? model.ExistingSelectedGks : model.TransferSelectedGks,
            Position.Defender => isExisting ? model.ExistingSelectedDefs : model.TransferSelectedDefs,
            Position.Midfielder => isExisting ? model.ExistingSelectedMids : model.TransferSelectedMids,
            Position.Forward => isExisting ? model.ExistingSelectedFwds : model.TransferSelectedFwds,
            _ => throw new ArgumentOutOfRangeException()
        };

        if (_request.Options.GetRidOf != null && player.Name == _request.Options.GetRidOf.Name)
        {
            model.Add(selection.SquadSelected == 0);
        }
        listOfPlayersToAddTo.Add(selection);

        return selection;
    }
}