using System.Net.Http.Json;
using System.Text.Json;
using FplTeamPicker.Domain.Contracts;
using FplTeamPicker.Domain.Models;
using FplTeamPicker.Services.Caching.Constants;
using FplTeamPicker.Services.Integrations.FplApi.Models;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.Logging;

namespace FplTeamPicker.Services.Integrations.FplApi;

public class FplRepository(
    HttpClient httpClient,
    JsonSerializerOptions serializerOptions,
    IFplUserProvider fplUserProvider,
    ILogger<FplRepository> logger,
    IMemoryCache memoryCache) : IFplRepository, IDisposable
{
    public async Task<User> GetUserDetails(CancellationToken cancellationToken)
    {
        var request = new HttpRequestMessage(HttpMethod.Get, "api/me");
        var result = await MakeRequestAsync<ApiUserDetails>(request, cancellationToken);

        memoryCache.Set(CacheKeys.UserLookup(fplUserProvider.GetUserId()), result.User.Entry);

        return new User
        {
            FirstName = result.User.FirstName,
            LastName = result.User.LastName,
            Id = result.User.Entry
        };
    }

    public async Task<Team> GetTeam(CancellationToken cancellationToken)
    {
        var userId = await GetManagerIdAsync(cancellationToken);
        var request = new HttpRequestMessage(HttpMethod.Get, $"api/my-team/{userId}");
        var result = await MakeRequestAsync<ApiTeam>(request, cancellationToken);

        var collection = result.Picks.Select(async p =>
        {
            var playerDetails = await LookupPlayerAsync(p.Id, cancellationToken);
            return new Player
            {
                Position = p.Position,
                Id = p.Id,
                IsCaptain = p.IsCaptain,
                IsViceCaptain = p.IsViceCaptain,
                InitialPurchasePrice = p.PurchasePrice,
                SellingPrice = p.SellingPrice,
                FirstName = playerDetails.FirstName,
                SecondName = playerDetails.SecondName,
                XpNext = playerDetails.XpNext,
                XpThis = playerDetails.XpThis,
                PurchasePrice = playerDetails.Cost,
            };
        }).ToList();
        return new Team
        {
            Players = await Task.WhenAll(collection),
        };
    }

    private async Task<int> GetManagerIdAsync(CancellationToken cancellationToken)
    {
        var cacheKey = CacheKeys.UserLookup(fplUserProvider.GetUserId());
        if (!memoryCache.TryGetValue(cacheKey, out int managerId))
        {
            var request = new HttpRequestMessage(HttpMethod.Get, "api/me");
            var result = await MakeRequestAsync<ApiUserDetails>(request, cancellationToken);
            managerId = result.User.Entry;
            memoryCache.Set(cacheKey, managerId);
        }

        return managerId;
    }

    private async Task<ApiPlayerDetails> LookupPlayerAsync(int playerId, CancellationToken cancellationToken)
    {
        var cacheKey = CacheKeys.PlayerLookup(playerId);
        if (!memoryCache.TryGetValue(cacheKey, out ApiPlayerDetails? playerDetails))
        {
            var request = new HttpRequestMessage(HttpMethod.Get, "api/bootstrap-static");
            var result = await MakeRequestAsync<ApiDataDump>(request, cancellationToken);
            foreach (var player in result.Players)
            {
                memoryCache.Set(CacheKeys.PlayerLookup(player.Id), player);
            }
        }

        return memoryCache.TryGetValue(cacheKey, out playerDetails)
            ? playerDetails!
            : throw new Exception("Player not found");
    }

    private async Task<TApiModel> MakeRequestAsync<TApiModel>(HttpRequestMessage request,
        CancellationToken cancellationToken)
        where TApiModel : class
    {
        var response = await httpClient.SendAsync(request, cancellationToken);

        if (!response.IsSuccessStatusCode)
        {
            var errorMessage = await response.Content.ReadAsStringAsync(cancellationToken);
            logger.LogError("API Error: {StatusCode} {Message}", response.StatusCode, errorMessage);
            throw new Exception("API Request failed");
        }

        var result = await response.Content.ReadFromJsonAsync<TApiModel>(serializerOptions, cancellationToken);
        return result!;
    }

    public void Dispose()
    {
        httpClient.Dispose();
    }
}