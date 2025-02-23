using Google.OrTools.Sat;

namespace FplTeamPicker.Exceptions;

/// <summary>
/// Exception to be thrown for handles optimisation-related exceptions
/// </summary>
public class OptimisationException : Exception
{
    private OptimisationException(string message) : base(message)
    {
    }

    public static OptimisationException SubOptimalCpSolution(CpSolverStatus status) =>
        new ($"No optimal solution was found for the CP-SAT solver. Status: {status}");

    public static OptimisationException NoSolution()=>
        new ("No solution was found.");
}