using Audacia.Random.Extensions;
using FplTeamPicker.Domain.Models;

namespace FplTeamPicker.Tests.Optimisation.Builders;

public class FplPlayerBuilder(int team, Position position)
{
    private int _cost;
    private decimal _xpNext;

    public FplPlayerBuilder WithCost(int cost)
    {
        _cost = cost;
        return this;
    }

    public FplPlayerBuilder WithPredictedPoints(decimal predictedPoints)
    {
        _xpNext = predictedPoints;
        return this;
    }

    public Player Build()
    {
        return new Player
        {
            Id = new Random().Next(),
            FirstName = new Random().Forename(),
            SecondName = new Random().Surname(),
            Cost = _cost,
            Team = team,
            Position = position,
            XpNext = _xpNext
        };
    }
}