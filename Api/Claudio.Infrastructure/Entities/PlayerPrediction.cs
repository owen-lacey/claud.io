namespace Claudio.Infrastructure.Entities;

using MongoDB.Bson.Serialization.Attributes;

[BsonIgnoreExtraElements]
public class PlayerPrediction
{
  [BsonElement("id")]
  public string Id { get; set; }
  
  [BsonElement("team")]
  public string Team { get; set; }

  [BsonElement("position")]
  public string Position { get; set; }

  [BsonElement("current_price")]
  public double CurrentPrice { get; set; }

  [BsonElement("expected_points")]
  public decimal ExpectedPoints { get; set; }

  [BsonElement("expected_goals")]
  public double ExpectedGoals { get; set; }

  [BsonElement("expected_assists")]
  public double ExpectedAssists { get; set; }

  [BsonElement("expected_saves")]
  public double ExpectedSaves { get; set; }

  [BsonElement("clean_sheet_prob")]
  public double? CleanSheetProb { get; set; }

  [BsonElement("predicted_goals_conceded")]
  public double? PredictedGoalsConceded { get; set; }

  [BsonElement("expected_minutes")]
  public double ExpectedMinutes { get; set; }

  [BsonElement("minutes_probability")]
  public double MinutesProbability { get; set; }

  [BsonElement("minutes_category")]
  public string MinutesCategory { get; set; }

  [BsonElement("expected_bonus")]
  public double ExpectedBonus { get; set; }

  [BsonElement("bonus_prob_3")]
  public double BonusProb3 { get; set; }

  [BsonElement("bonus_prob_2")]
  public double BonusProb2 { get; set; }

  [BsonElement("bonus_prob_1")]
  public double BonusProb1 { get; set; }

  [BsonElement("points_per_million")]
  public double PointsPerMillion { get; set; }

  [BsonElement("value_rank")]
  public int ValueRank { get; set; }

  [BsonElement("form_adjustment")]
  public double FormAdjustment { get; set; }

  [BsonElement("prediction_confidence")]
  public double PredictionConfidence { get; set; }

  [BsonElement("variance")]
  public double Variance { get; set; }

  [BsonElement("ceiling")]
  public double Ceiling { get; set; }

  [BsonElement("floor")]
  public double Floor { get; set; }

  [BsonElement("fixture_difficulty")]
  public int FixtureDifficulty { get; set; }

  [BsonElement("home_away")]
  public string HomeAway { get; set; }

  [BsonElement("opponent")]
  public string Opponent { get; set; }

  [BsonElement("opponent_strength_attack")]
  public int OpponentStrengthAttack { get; set; }

  [BsonElement("opponent_strength_defence")]
  public int OpponentStrengthDefence { get; set; }

  [BsonElement("fixture_attractiveness")]
  public double FixtureAttractiveness { get; set; }

  [BsonElement("xp")]
  public decimal Xp { get; set; }
}