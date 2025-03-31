using FplTeamPicker.Api.Providers.Exceptions;
using Microsoft.AspNetCore.Diagnostics;

namespace FplTeamPicker.Api.Exceptions;

public class ExceptionHandler : IExceptionHandler
{
    public ValueTask<bool> TryHandleAsync(HttpContext httpContext, Exception exception,
        CancellationToken cancellationToken)
    {
        if (exception is SecurityException)
        {
            httpContext.Response.StatusCode = StatusCodes.Status401Unauthorized;
            return ValueTask.FromResult(true);
        }

        return ValueTask.FromResult(false);
    }
}