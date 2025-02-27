namespace FplTeamPicker.Services.Caching.Constants;

public static class CacheKeys
{
    public static string UserLookup(int userId) => $"user-{userId}";
    
    public static string PlayerLookup(int playerId) => $"player-{playerId}";
}