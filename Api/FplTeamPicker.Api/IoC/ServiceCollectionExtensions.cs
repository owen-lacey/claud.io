using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using FplTeamPicker.Domain.Contracts;
using FplTeamPicker.Services.Integrations.FplApi;
using FplTeamPicker.Services.Integrations.FplApi.Constants;

namespace FplTeamPicker.Api.IoC;

public static class ServiceCollectionExtensions
{
    public static IServiceCollection AddFplApi(this IServiceCollection services)
    {
        services.AddHttpClient<IFplRepository, FplRepository>((serviceProvider, client) =>
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
        services.AddSingleton(new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
        });
        return services;
    }
}