namespace FplTeamPicker.Services.Caching.Constants;

public static class CacheKeys
{
    public const string CurrentGameweek = "current-gw";
    
    public const string Teams = "teams";
    
    public const string Players = "players";
    
    public const string Managers = "managers";
    
    public static string PlayerLookup(int playerId) => $"player-{playerId}";
    
    public static string TeamLookup(int team) => $"team-{team}";
}