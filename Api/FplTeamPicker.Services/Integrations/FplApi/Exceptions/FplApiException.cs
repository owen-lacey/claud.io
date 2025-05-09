using System.Net;

namespace FplTeamPicker.Services.Integrations.FplApi.Exceptions;

public class FplApiException(HttpStatusCode responseStatusCode, string errorMessage)
    : Exception($"FPL API message: {errorMessage}")
{
    public HttpStatusCode ResponseStatusCode { get; } = responseStatusCode;
    
    public string ErrorMessage { get; } = errorMessage;
}