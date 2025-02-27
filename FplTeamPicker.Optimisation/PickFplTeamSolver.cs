using FplTeamPicker.Optimisation.Exceptions;
using FplTeamPicker.Optimisation.Models;
using Google.OrTools.Sat;

namespace FplTeamPicker.Optimisation;

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
        var index = 0;
        foreach (var fplPlayer in _request.Players)
        {
            var squadSelected = cpSolver.BooleanValue(model.Selections[index].SquadSelected);
            var teamSelected = cpSolver.BooleanValue(model.Selections[index].TeamSelected);
            if (squadSelected)
            {
                output.Squad.Add(fplPlayer);
            }
            if (teamSelected)
            {
                output.Team.Add(fplPlayer);
            }

            index++;
        }

        return output;
    }

    private void InitialiseVars(PickFplTeamCpModel model)
    {
        foreach (var fplPlayer in _request.Players)
        {
            var squadSelection = GetSelectionVar(model, fplPlayer);

            if (model.TeamSelectionCounts.ContainsKey(fplPlayer.Team))
            {
                model.TeamSelectionCounts[fplPlayer.Team].Add(squadSelection.SquadSelected);
            }
            else
            {
                model.TeamSelectionCounts.Add(fplPlayer.Team, [squadSelection.SquadSelected]);
            }
        }
    }

    private void AddObjective(PickFplTeamCpModel model)
    {
        // Get the scale product of the selection booleans and multiply by each player's XI selection, maximizing the total.
        var allPlayerPredictedPoints =
            _request.Players.Select(p => (int)Math.Round(p.PredictedPoints * 100)).ToList();
        var teamPredictedPoints = LinearExpr.ScalProd(model.Selections.Select(s => s.SquadSelected), allPlayerPredictedPoints);
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

    private void AddPositionConstraints(PickFplTeamCpModel model)
    {
        // Sum up the selections for each position, and ensure it matches the number of players required per position.
        model.Add(new SumArray(model.SelectedGks.Select(p => p.SquadSelected)) == _request.Options.SquadGoalkeeperCount);
        model.Add(new SumArray(model.SelectedDefs.Select(p => p.SquadSelected)) == _request.Options.SquadDefenderCount);
        model.Add(new SumArray(model.SelectedMids.Select(p => p.SquadSelected)) == _request.Options.SquadMidfielderCount);
        model.Add(new SumArray(model.SelectedFwds.Select(p => p.SquadSelected)) == _request.Options.SquadForwardCount);
    }

    private void AddMaxPerTeamConstraint(PickFplTeamCpModel model)
    {
        // Sum up the selections for each team, and ensure it doesn't exceed the maximum number of players per team.
        foreach (var (_, selections) in model.TeamSelectionCounts)
        {
            model.Add(new SumArray(selections) <= _request.Options.MaxPlayersPerTeam);
        }
    }

    private void AddCostConstraint(PickFplTeamCpModel model)
    {
        // Get the scale product of the selection booleans and multiply by each player's cost.
        var allPlayerCosts = _request.Players.Select(p => p.Cost).ToList();
        var allPlayerSelections = LinearExpr.ScalProd(model.Selections.Select(p => p.SquadSelected), allPlayerCosts);
        model.Add(allPlayerSelections <= _request.Budget);
    }

    private static FplPlayerSelectionVar GetSelectionVar(PickFplTeamCpModel model, FplPlayer fplPlayer)
    {
        var selection = new FplPlayerSelectionVar(model, fplPlayer);

        switch (fplPlayer.Position)
        {
            case PlayerPosition.GK:
                model.SelectedGks.Add(selection);
                break;
            case PlayerPosition.DEF:
                model.SelectedDefs.Add(selection);
                break;
            case PlayerPosition.MID:
                model.SelectedMids.Add(selection);
                break;
            case PlayerPosition.FWD:
                model.SelectedFwds.Add(selection);
                break;
        }

        return selection;
    }
}