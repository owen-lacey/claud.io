using MongoDB.Driver;
using Claudio.Infrastructure.Entities;

namespace Claudio.Infrastructure;

public class ClaudioContext
{
    private readonly IMongoDatabase _database;

    public ClaudioContext(string connectionString, string databaseName)
    {
        var client = new MongoClient(connectionString);
        _database = client.GetDatabase(databaseName);
    }

    public IMongoCollection<Fixture> Fixtures => _database.GetCollection<Fixture>("fixtures");
    public IMongoCollection<Team> Teams => _database.GetCollection<Team>("teams");
    public IMongoCollection<Player> Players => _database.GetCollection<Player>("players");
}