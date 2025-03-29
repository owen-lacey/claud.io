using Audacia.Random.Extensions;
using FplTeamPicker.Domain.Models;

namespace FplTeamPicker.Tests.Optimisation.Builders;

public class FplPlayerBuilder
{
    private readonly int _team;
    private readonly Position _position;
    private int _cost;
    private decimal _xpNext;

    public FplPlayerBuilder(int team, Position position)
    {
        _team = team;
        _position = position;
    }

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
            Team = _team,
            Position = _position,
            XpNext = _xpNext
        };
    }
}