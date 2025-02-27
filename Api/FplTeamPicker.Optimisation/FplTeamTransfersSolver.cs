using FplTeamPicker.Optimisation.Exceptions;
using FplTeamPicker.Optimisation.Models;
using Google.OrTools.Sat;

namespace FplTeamPicker.Optimisation;

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
        var index = 0;
        foreach (var fplPlayer in _request.ExistingPlayers)
        {
            var squadSelected = cpSolver.BooleanValue(model.Selections[index].SquadSelected);
            var teamSelected = cpSolver.BooleanValue(model.Selections[index].TeamSelected);
            if (squadSelected)
            {
                output.PlayersKept.Add(fplPlayer);
            }
            else
            {
                output.PlayersOut.Add(fplPlayer);
            }

            if (teamSelected)
            {
                output.Team.Add(fplPlayer);
            }

            index++;
        }

        index = 0;
        foreach (var fplPlayer in _request.OtherPlayers)
        {
            var squadSelected = cpSolver.BooleanValue(model.TransferSelections[index].SquadSelected);
            var teamSelected = cpSolver.BooleanValue(model.TransferSelections[index].TeamSelected);
            if (squadSelected)
            {
                output.Squad.Add(fplPlayer);
                output.PlayersIn.Add(fplPlayer);
            }

            if (teamSelected)
            {
                output.Team.Add(fplPlayer);
            }

            index++;
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

    private static void InitTeamSelectionCount(FplTeamTransfersCpModel model, FplPlayer existingPlayer,
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
        var allPlayerPredictedPoints = _request.AllPlayers.Select(p => (int)Math.Round(p.PredictedPoints * 100)).ToList();
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
        model.Add(teamDefenders <= _request.Options.SquadDefenderCount);
        model.Add(teamMidfielders >= _request.Options.MinTeamMidfielders);
        model.Add(teamMidfielders <= _request.Options.SquadMidfielderCount);
        model.Add(teamForwards >= _request.Options.MinTeamForwards);
        model.Add(teamForwards <= _request.Options.SquadForwardCount);

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

    private FplPlayerSelectionVar GetSelectionVar(FplTeamTransfersCpModel model, FplPlayer fplPlayer, bool isExisting)
    {
        var selection = new FplPlayerSelectionVar(model, fplPlayer);

        List<FplPlayerSelectionVar> listOfPlayersToAddTo = fplPlayer.Position switch
        {
            PlayerPosition.GK => isExisting ? model.ExistingSelectedGks : model.TransferSelectedGks,
            PlayerPosition.DEF => isExisting ? model.ExistingSelectedDefs : model.TransferSelectedDefs,
            PlayerPosition.MID => isExisting ? model.ExistingSelectedMids : model.TransferSelectedMids,
            PlayerPosition.FWD => isExisting ? model.ExistingSelectedFwds : model.TransferSelectedFwds,
            _ => throw new ArgumentOutOfRangeException()
        };

        if (_request.Options.GetRidOf != null && fplPlayer.Name == _request.Options.GetRidOf.Name)
        {
            model.Add(selection.SquadSelected == 0);
        }
        listOfPlayersToAddTo.Add(selection);

        return selection;
    }
}