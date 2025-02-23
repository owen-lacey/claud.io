using Audacia.Random.Extensions;
using FplTeamPicker.Models;

namespace FplTeamPicker.Tests.Builders;

public class FplPlayerBuilder(string team, PlayerPosition position)
{
    private int _cost;
    private decimal _predictedPoints;

    public FplPlayerBuilder WithCost(int cost)
    {
        _cost = cost;
        return this;
    }

    public FplPlayerBuilder WithPredictedPoints(decimal predictedPoints)
    {
        _predictedPoints = predictedPoints;
        return this;
    }

    public FplPlayer Build()
    {
        return new FplPlayer
        {
            Name = new Random().Surname(),
            Cost = _cost,
            Team = team,
            Position = position,
            PredictedPoints = _predictedPoints
        };
    }
}