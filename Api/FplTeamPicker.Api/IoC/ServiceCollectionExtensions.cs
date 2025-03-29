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
                    out var cookie) == true)
            {
                client.DefaultRequestHeaders.Add("Cookie", $"{FplApiConstants.HeaderName}={cookie}");
            }

            const string fplTeamApiUrl = "https://fantasy.premierleague.com";
            client.BaseAddress = new Uri(fplTeamApiUrl);
        });
        services.AddSingleton(new JsonSerializerOptions
        {
            PropertyNamingPolicy = new SnakeCaseNamingPolicy(),
        });
        return services;
    }
}

public class SnakeCaseNamingPolicy : JsonNamingPolicy
{
    public override string ConvertName(string name) =>
        name.ToSnakeCase();
}

public static class Foo
{
    internal enum SnakeCaseState
    {
        Start,
        Lower,
        Upper,
        NewWord
    }

    public static string ToSnakeCase(this string s)
    {
        if (string.IsNullOrEmpty(s))
        {
            return s;
        }

        StringBuilder sb = new StringBuilder();
        SnakeCaseState state = SnakeCaseState.Start;

        for (int i = 0; i < s.Length; i++)
        {
            if (s[i] == ' ')
            {
                if (state != SnakeCaseState.Start)
                {
                    state = SnakeCaseState.NewWord;
                }
            }
            else if (char.IsUpper(s[i]))
            {
                switch (state)
                {
                    case SnakeCaseState.Upper:
                        bool hasNext = (i + 1 < s.Length);
                        if (i > 0 && hasNext)
                        {
                            char nextChar = s[i + 1];
                            if (!char.IsUpper(nextChar) && nextChar != '_')
                            {
                                sb.Append('_');
                            }
                        }

                        break;
                    case SnakeCaseState.Lower:
                    case SnakeCaseState.NewWord:
                        sb.Append('_');
                        break;
                }

                char c;
#if HAVE_CHAR_TO_LOWER_WITH_CULTURE
                    c = char.ToLower(s[i], CultureInfo.InvariantCulture);
#else
                c = char.ToLowerInvariant(s[i]);
#endif
                sb.Append(c);

                state = SnakeCaseState.Upper;
            }
            else if (s[i] == '_')
            {
                sb.Append('_');
                state = SnakeCaseState.Start;
            }
            else
            {
                if (state == SnakeCaseState.NewWord)
                {
                    sb.Append('_');
                }

                sb.Append(s[i]);
                state = SnakeCaseState.Lower;
            }
        }

        return sb.ToString();
    }
}