using FplTeamPicker.Domain.Models;
using MongoDB.Bson;
using MongoDB.Bson.Serialization.Attributes;

namespace Claudio.Infrastructure.Entities;

public class Player
{
  [BsonId]
  [BsonRepresentation(BsonType.ObjectId)]
  public string? MongoId { get; set; }

  [BsonElement("id")]
  public int Id { get; set; }

  [BsonElement("code")]
  public int Code { get; set; }

  [BsonElement("web_name")]
  public string WebName { get; set; } = string.Empty;

  [BsonElement("first_name")]
  public string FirstName { get; set; } = string.Empty;

  [BsonElement("second_name")]
  public string SecondName { get; set; } = string.Empty;

  [BsonElement("element_type")]
  public Position ElementType { get; set; }

  [BsonElement("team")]
  public int Team { get; set; }

  [BsonElement("now_cost")]
  public int NowCost { get; set; }

  [BsonElement("selected_by_percent")]
  public decimal SelectedByPercent { get; set; }

  [BsonElement("form")]
  public decimal Form { get; set; }

  [BsonElement("points_per_game")]
  public decimal PointsPerGame { get; set; }

  [BsonElement("total_points")]
  public int TotalPoints { get; set; }

  [BsonElement("minutes")]
  public int Minutes { get; set; }

  [BsonElement("goals_scored")]
  public int GoalsScored { get; set; }

  [BsonElement("assists")]
  public int Assists { get; set; }

  [BsonElement("clean_sheets")]
  public int CleanSheets { get; set; }

  [BsonElement("goals_conceded")]
  public int GoalsConceded { get; set; }

  [BsonElement("own_goals")]
  public int OwnGoals { get; set; }

  [BsonElement("penalties_saved")]
  public int PenaltiesSaved { get; set; }

  [BsonElement("penalties_missed")]
  public int PenaltiesMissed { get; set; }

  [BsonElement("yellow_cards")]
  public int YellowCards { get; set; }

  [BsonElement("red_cards")]
  public int RedCards { get; set; }

  [BsonElement("saves")]
  public int Saves { get; set; }

  [BsonElement("bonus")]
  public int Bonus { get; set; }

  [BsonElement("bps")]
  public int Bps { get; set; }

  [BsonElement("expected_goals")]
  public decimal ExpectedGoals { get; set; }

  [BsonElement("expected_assists")]
  public decimal ExpectedAssists { get; set; }

  [BsonElement("expected_goal_involvements")]
  public decimal ExpectedGoalInvolvements { get; set; }

  [BsonElement("expected_goals_conceded")]
  public decimal ExpectedGoalsConceded { get; set; }

  [BsonElement("predictions")]
  public Dictionary<int, PlayerPrediction> Predictions { get; set; }

  public FplTeamPicker.Domain.Models.Player ToPlayer(int currentGameweek)
  {
    return new FplTeamPicker.Domain.Models.Player
    {
      Id = Code,
      Position = (Position)ElementType,
      Cost = NowCost,
      ChanceOfPlayingNextRound = null, // Not available in entity
      FirstName = FirstName,
      SecondName = SecondName,
      Xp = null, // Not available at root level in entity
      SelectedByPercent = SelectedByPercent,
      Team = Team,
      SeasonPoints = TotalPoints,
      YellowCards = YellowCards,
      BirthDate = null, // Not available in entity
      RedCards = RedCards,
      TransfersOut = 0, // Not available in entity
      Predictions = Predictions != null
            ? Predictions.ToDictionary(
                kvp => kvp.Key,
                kvp => kvp.Value.ExpectedPoints)
            : new Dictionary<int, decimal>()
    };
  }
}