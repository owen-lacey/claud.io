using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Claudio.Infrastructure;
using FplTeamPicker.Domain.Contracts;
using FplTeamPicker.Services.Integrations.FplApi;
using FplTeamPicker.Services.Integrations.FplApi.Constants;

namespace FplTeamPicker.Api.IoC;

public static class ServiceCollectionExtensions
{
    public static IServiceCollection AddFplApi(this IServiceCollection services)
    {
        services.AddHttpClient<IReferenceDataRepository, ReferenceDataRepository>((serviceProvider, client) =>
        {
            // pass through the cookie from the client to the FPL API
            var httpContextAccessor = serviceProvider.GetRequiredService<IHttpContextAccessor>();
            if (httpContextAccessor.HttpContext?.Request.Headers.TryGetValue(FplApiConstants.HeaderName,
                    out var token) == true)
            {
                client.DefaultRequestHeaders.Add("X-Api-Authorization", $"Bearer {token}");
            }

            const string fplTeamApiUrl = "https://fantasy.premierleague.com";
            client.BaseAddress = new Uri(fplTeamApiUrl);
        });
        services.AddHttpClient<IUserRepository, UserRepository>((serviceProvider, client) =>
        {
            // pass through the cookie from the client to the FPL API
            var httpContextAccessor = serviceProvider.GetRequiredService<IHttpContextAccessor>();
            if (httpContextAccessor.HttpContext?.Request.Headers.TryGetValue(FplApiConstants.HeaderName,
                    out var token) == true)
            {
                client.DefaultRequestHeaders.Add("X-Api-Authorization", $"Bearer {token}");
            }

            const string fplTeamApiUrl = "https://fantasy.premierleague.com";
            client.BaseAddress = new Uri(fplTeamApiUrl);
        });
        services.AddScoped<ClaudioContext>((_) =>
        {
            var context = new ClaudioContext(
                "mongodb://admin:fplteampicker2025@localhost:27017/",
                "fpl_team_picker"
            );
            return context;
        });
        services.AddSingleton(new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
        });
        return services;
    }
}