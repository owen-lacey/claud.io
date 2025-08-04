using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using Claudio.Infrastructure;
using FplTeamPicker.Domain.Contracts;
using FplTeamPicker.Domain.Models;
using FplTeamPicker.Services.Caching.Constants;
using FplTeamPicker.Services.Integrations.FplApi.Exceptions;
using FplTeamPicker.Services.Integrations.FplApi.Models;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.Logging;
using MongoDB.Driver;
using MongoDB.Driver.Linq;

namespace FplTeamPicker.Services.Integrations.FplApi;

public class UserRepository : IUserRepository, IDisposable
{
    private readonly HttpClient _httpClient;
    private readonly JsonSerializerOptions _serializerOptions;
    private readonly ILogger<ReferenceDataRepository> _logger;
    private readonly ClaudioContext _claudioContext;
    private readonly IMemoryCache _memoryCache;

    public UserRepository(
        HttpClient httpClient,
        JsonSerializerOptions serializerOptions,
        ILogger<ReferenceDataRepository> logger,
        ClaudioContext claudioContext,
        IMemoryCache memoryCache)
    {
        _httpClient = httpClient;
        _serializerOptions = serializerOptions;
        _logger = logger;
        _claudioContext = claudioContext;
        _memoryCache = memoryCache;
    }

    public async Task<User> GetUserDetailsAsync(CancellationToken cancellationToken)
    {
        var request = new HttpRequestMessage(HttpMethod.Get, "api/me");
        var result = await MakeRequestAsync<ApiUserDetails>(request, cancellationToken);

        if (result.User == null)
        {
            throw new FplApiException(HttpStatusCode.Unauthorized, $"Unable to get user details.");
        }

        return new User
        {
            FirstName = result.User.FirstName,
            LastName = result.User.LastName,
            Id = result.User.Entry
        };
    }

    public async Task<MyTeam> GetMyTeamAsync(CancellationToken cancellationToken)
    {
        var userId = await GetManagerIdAsync(cancellationToken);
        var teamRequest = new HttpRequestMessage(HttpMethod.Get, $"api/my-team/{userId}");
        var teamResult = await MakeRequestAsync<ApiTeam>(teamRequest, cancellationToken);
        var selectedTeam = new SelectedSquad();
        var team = new MyTeam
        {
            Bank = teamResult.Transfers.Bank,
            FreeTransfers = (teamResult.Transfers.Limit ?? 0) - teamResult.Transfers.Made,
            SelectedSquad = selectedTeam
        };

        foreach (var pick in teamResult.Picks
                     .Where(p => p.Position != (int)Position.Manager))
        {
            var playerDetails = await LookupPlayerAsync(pick.Id, cancellationToken);
            var selectedPlayer = new SelectedPlayer
            {
                IsCaptain = pick.IsCaptain,
                IsViceCaptain = pick.IsViceCaptain,
                Player = playerDetails,
                SellingPrice = pick.SellingPrice
            };

            if (pick.SquadNumber <= 11)
            {
                selectedTeam.StartingXi.Add(selectedPlayer);
            }
            else
            {
                selectedTeam.Bench.Add(selectedPlayer);
            }
        }

        return team;
    }

    public async Task<SelectedSquad> GetSelectedTeamAsync(int userId, int gameweek, CancellationToken cancellationToken)
    {
        var request = new HttpRequestMessage(HttpMethod.Get, $"api/entry/{userId}/event/{gameweek}/picks");
        var result = await MakeRequestAsync<ApiEntryPicks>(request, cancellationToken);

        var team = new SelectedSquad();
        foreach (var pick in result.Picks
                     .Where(p => p.Position != (int)Position.Manager))
        {
            var playerDetails = await LookupPlayerAsync(pick.Id, cancellationToken);
            var selectedPlayer = new SelectedPlayer
            {
                IsCaptain = pick.IsCaptain,
                IsViceCaptain = pick.IsViceCaptain,
                Player = playerDetails,
                SellingPrice = pick.SellingPrice
            };

            if (pick.SquadNumber <= 11)
            {
                team.StartingXi.Add(selectedPlayer);
            }
            else
            {
                team.Bench.Add(selectedPlayer);
            }
        }

        return team;
    }

    public async Task<List<League>> GetLeaguesAsync(CancellationToken cancellationToken)
    {
        var userId = await GetManagerIdAsync(cancellationToken);
        var request = new HttpRequestMessage(HttpMethod.Get, $"api/entry/{userId}");
        var result = await MakeRequestAsync<ApiEntry>(request, cancellationToken);
        var leagues = new List<League>();
        foreach (var classicLeague in result.Leagues.Classic.Where(c => c.LeagueType == "x"))
        {
            var leagueRequest =
                new HttpRequestMessage(HttpMethod.Get, $"api/leagues-classic/{classicLeague.Id}/standings");
            var leagueResult = await MakeRequestAsync<ApiLeagueDetails>(leagueRequest, cancellationToken);
            if (leagueResult.Standings.HasNext)
            {
                throw new Exception("Lots of players in this league, help!");
            }

            var league = new League
            {
                Id = classicLeague.Id,
                Name = classicLeague.Name,
                CurrentPosition = classicLeague.EntryRank,
                Participants = leagueResult.Standings.Results
                    .Select(r => new LeagueParticipant
                    {
                        UserId = r.Entry,
                        PlayerName = r.PlayerName,
                        TeamName = r.TeamName,
                        Position = r.Rank,
                        Total = r.Total
                    })
                    .ToList()
            };
            leagues.Add(league);
        }

        return leagues;
    }

    private async Task<int> GetManagerIdAsync(CancellationToken cancellationToken)
    {
        var request = new HttpRequestMessage(HttpMethod.Get, "api/me");
        var result = await MakeRequestAsync<ApiUserDetails>(request, cancellationToken);
        var managerId = result.User?.Entry ?? throw new FplApiException(HttpStatusCode.Unauthorized, "Unable to get user details");

        return managerId;
    }

    private async Task<Player> LookupPlayerAsync(int playerId, CancellationToken cancellationToken)
    {
        var currentGameweek = await GetCurrentGameweekAsync(cancellationToken);
        var player = await _claudioContext.Players
            .AsQueryable()
            .SingleOrDefaultAsync(p => p.Id == playerId, cancellationToken);
        return player.ToPlayer(currentGameweek);
    }

    private async Task<TApiModel> MakeRequestAsync<TApiModel>(HttpRequestMessage request,
        CancellationToken cancellationToken)
        where TApiModel : class
    {
        var response = await _httpClient.SendAsync(request, cancellationToken);

        if (!response.IsSuccessStatusCode)
        {
            var errorMessage = await response.Content.ReadAsStringAsync(cancellationToken);
            _logger.LogError("API Error: {StatusCode} {Message}", response.StatusCode, errorMessage);
            throw new FplApiException(response.StatusCode, errorMessage);
        }

        var result = await response.Content.ReadFromJsonAsync<TApiModel>(_serializerOptions, cancellationToken);
        return result!;
    }

    private async Task<int> GetCurrentGameweekAsync(CancellationToken cancellationToken)
    {
        if (!_memoryCache.TryGetValue(CacheKeys.CurrentGameweek, out _))
        {
            await DoBulkDataLoadAsync(cancellationToken);
        }

        return _memoryCache.Get<int>(CacheKeys.CurrentGameweek);
    }

    private async Task DoBulkDataLoadAsync(CancellationToken cancellationToken)
    {
        var request = new HttpRequestMessage(HttpMethod.Get, "api/bootstrap-static");
        var result = await MakeRequestAsync<ApiDataDump>(request, cancellationToken);

        var currentGameweek = result.Events.SingleOrDefault(e => e.IsCurrent)
                              ?? result.Events.Single(e => e.IsNext);
        _memoryCache.Set(CacheKeys.CurrentGameweek, currentGameweek.Id);
    }


    public void Dispose()
    {
        _httpClient.Dispose();
    }
}