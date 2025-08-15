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

public class ReferenceDataRepository : IReferenceDataRepository, IDisposable
{
  private readonly HttpClient _httpClient;
  private readonly JsonSerializerOptions _serializerOptions;
  private readonly ILogger<ReferenceDataRepository> _logger;
  private readonly IMemoryCache _memoryCache;
  private readonly ClaudioContext _claudioContext;

  public ReferenceDataRepository(
      HttpClient httpClient,
      JsonSerializerOptions serializerOptions,
      ILogger<ReferenceDataRepository> logger,
      IMemoryCache memoryCache,
      ClaudioContext claudioContext)
  {
    _httpClient = httpClient;
    _serializerOptions = serializerOptions;
    _logger = logger;
    _memoryCache = memoryCache;
    _claudioContext = claudioContext;
  }

  public async Task<List<Player>> GetPlayersAsync(CancellationToken cancellationToken)
  {
    var currentGameweek = await GetCurrentGameweekAsync(cancellationToken);
    var players = await _claudioContext.Players.AsQueryable()
        .ToListAsync(cancellationToken);

    return players.Select(p => p.ToPlayer(currentGameweek)).ToList();
  }

  public async Task<List<Team>> GetTeamsAsync(CancellationToken cancellationToken)
  {
    var teams = await _claudioContext.Teams.AsQueryable().ToListAsync();
    var currentGameweek = await GetCurrentGameweekAsync(cancellationToken);
    return teams.Select(t => t.ToTeam(currentGameweek)).ToList();
  }

  public async Task<List<Fixture>> GetFixturesAsync(CancellationToken cancellationToken)
  {
    var fixtures = await _claudioContext.Fixtures.AsQueryable()
        .ToListAsync(cancellationToken);

    return fixtures.Select(f => f.ToFixture()).ToList();
  }

  public async Task<int> GetCurrentGameweekAsync(CancellationToken cancellationToken)
  {
    if (!_memoryCache.TryGetValue(CacheKeys.CurrentGameweek, out _))
    {
      await DoBulkDataLoadAsync(cancellationToken);
    }

    return _memoryCache.Get<int>(CacheKeys.CurrentGameweek);
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