using FplTeamPicker.Domain.Models;
using MongoDB.Bson;
using MongoDB.Bson.Serialization.Attributes;

namespace Claudio.Infrastructure.Entities;

public class Team
{
    [BsonId]
    [BsonRepresentation(BsonType.ObjectId)]
    public string? MongoId { get; set; }
    
    [BsonElement("id")]
    public int Id { get; set; }

    [BsonElement("fbref_id")]
    public string FbRefId { get; set; } = null!;
    
    [BsonElement("name")]
    public string Name { get; set; } = string.Empty;
    
    [BsonElement("short_name")]
    public string ShortName { get; set; } = string.Empty;
    
    [BsonElement("code")]
    public int Code { get; set; }
    
    [BsonElement("strength")]
    public int Strength { get; set; }
    
    [BsonElement("strength_overall_home")]
    public int StrengthOverallHome { get; set; }
    
    [BsonElement("strength_overall_away")]
    public int StrengthOverallAway { get; set; }
    
    [BsonElement("strength_attack_home")]
    public int StrengthAttackHome { get; set; }
    
    [BsonElement("strength_attack_away")]
    public int StrengthAttackAway { get; set; }
    
    [BsonElement("strength_defence_home")]
    public int StrengthDefenceHome { get; set; }
    
    [BsonElement("strength_defence_away")]
    public int StrengthDefenceAway { get; set; }
    
    [BsonElement("played")]
    public int Played { get; set; }
    
    [BsonElement("win")]
    public int Win { get; set; }
    
    [BsonElement("draw")]
    public int Draw { get; set; }
    
    [BsonElement("loss")]
    public int Loss { get; set; }
    
    [BsonElement("points")]
    public int Points { get; set; }
    
    [BsonElement("position")]
    public int Position { get; set; }

    [BsonElement("strengths")]
    public Dictionary<int, TeamStrength> Strengths { get; set; } = new();

    public FplTeamPicker.Domain.Models.Team ToTeam(int currentGameweek)
    {
        return new FplTeamPicker.Domain.Models.Team
        {
            ShortName = ShortName,
            Name = Name,
            Code = Code,
            Id = Id,
            CurrentStrength = Strengths.TryGetValue(currentGameweek, out var strength)
                ? strength.Overall
                : null
        };
    }
}
