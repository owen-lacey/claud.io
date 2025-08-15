using MongoDB.Bson.Serialization.Attributes;

namespace Claudio.Infrastructure.Entities;

public class TeamStrength
{
    [BsonElement("attack")]
    public double Attack { get; set; }
    
    [BsonElement("defence")]
    public double Defence { get; set; }
    
    [BsonElement("overall")]
    public double Overall { get; set; }
}