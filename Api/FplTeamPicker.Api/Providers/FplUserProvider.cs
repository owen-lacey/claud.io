using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using FplTeamPicker.Api.Providers.Exceptions;
using FplTeamPicker.Domain.Contracts;
using FplTeamPicker.Services.Integrations.FplApi.Constants;

namespace FplTeamPicker.Api.Providers;

public class FplUserProvider(IHttpContextAccessor httpContextAccessor) : IFplUserProvider
{
    public record FplCookie
    {
        [JsonPropertyName("u")]
        public FplCookieUser User { get; set; }
    }

    public record FplCookieUser
    {
        [JsonPropertyName("id")]
        public int Id { get; set; }
    }
    public int GetUserId()
    {
        int? userId;
        if (httpContextAccessor.HttpContext?.Request.Cookies.TryGetValue(
                FplApiConstants.CookieName,
                out var cookieStr) == true)
        {
            var decodedCookieString = Convert.FromBase64String(cookieStr);
            var cookieJson = Encoding.UTF8.GetString(decodedCookieString);
            var cookieBytes = JsonSerializer.Deserialize<FplCookie>(cookieJson);

            userId = cookieBytes?.User.Id;
            return userId ?? throw new SecurityException(
                $"Invalid cookie provided. Raw cookie length: {cookieStr}. Decoded cookie length: {decodedCookieString}");
        }

        throw new SecurityException($"No cookie with the name {FplApiConstants.CookieName} was provided");
    }
}