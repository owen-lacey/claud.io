using MongoDB.Bson;
using MongoDB.Bson.Serialization.Attributes;

namespace Claudio.Infrastructure.Entities;

public class Fixture
{
    [BsonId]
    [BsonRepresentation(BsonType.ObjectId)]
    public string? MongoId { get; set; }
    
    [BsonElement("id")]
    public int Id { get; set; }
    
    [BsonElement("gameweek")]
    public int Gameweek { get; set; }
    
    [BsonElement("season")]
    public string Season { get; set; } = string.Empty;
    
    [BsonElement("team_h")]
    public int TeamHome { get; set; }
    
    [BsonElement("team_a")]
    public int TeamAway { get; set; }
    
    [BsonElement("team_h_difficulty")]
    public int TeamHomeDifficulty { get; set; }
    
    [BsonElement("team_a_difficulty")]
    public int TeamAwayDifficulty { get; set; }
    
    [BsonElement("kickoff_time")]
    [BsonDateTimeOptions(Kind = DateTimeKind.Utc)]
    public DateTime KickoffTime { get; set; }
    
    [BsonElement("finished")]
    public bool Finished { get; set; }
    
    [BsonElement("team_h_score")]
    public int? TeamHomeScore { get; set; }
    
    [BsonElement("team_a_score")]
    public int? TeamAwayScore { get; set; }
}
