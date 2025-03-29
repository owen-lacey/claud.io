using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using FplTeamPicker.Api.Providers.Exceptions;
using FplTeamPicker.Domain.Contracts;
using FplTeamPicker.Services.Integrations.FplApi.Constants;

namespace FplTeamPicker.Api.Providers;

public class FplUserProvider : IFplUserProvider
{
    private readonly IHttpContextAccessor _httpContextAccessor;

    public FplUserProvider(IHttpContextAccessor httpContextAccessor)
    {
        _httpContextAccessor = httpContextAccessor;
    }
    
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
        if (_httpContextAccessor.HttpContext?.Request.Headers.TryGetValue(
                FplApiConstants.HeaderName,
                out var headerStr) == true)
        {
            var s = headerStr.ToString();
            var decodedCookieString = Convert.FromBase64String(s);
            var cookieJson = Encoding.UTF8.GetString(decodedCookieString);
            var cookieBytes = JsonSerializer.Deserialize<FplCookie>(cookieJson);

            userId = cookieBytes?.User.Id;
            return userId ?? throw new SecurityException(
                $"Invalid cookie provided. Raw cookie length: {headerStr}. Decoded cookie length: {decodedCookieString}");
        }

        throw new SecurityException($"No header with the name {FplApiConstants.HeaderName} was provided");
    }
}