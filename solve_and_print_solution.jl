# from JuMP.jl docs
function solve_and_print_solution(model)
    optimize!(model)
    status = termination_status(model)
    if status in (OPTIMAL, LOCALLY_SOLVED)
        println("Solution is optimal")
    elseif status in (ALMOST_OPTIMAL, ALMOST_LOCALLY_SOLVED)
        println("Solution is optimal to a relaxed tolerance")
    elseif status == TIME_LIMIT
        println(
                "Solver stopped due to a time limit. If a solution is available, " *
                "it may be suboptimal."
               )
    elseif status in (
                      ITERATION_LIMIT, NODE_LIMIT, SOLUTION_LIMIT, MEMORY_LIMIT,
                      OBJECTIVE_LIMIT, NORM_LIMIT, OTHER_LIMIT,
                     )
        println(
                "Solver stopped due to a limit. If a solution is available, it " *
                "may be suboptimal."
               )
    elseif status in (INFEASIBLE, LOCALLY_INFEASIBLE)
        println("The problem is primal infeasible")
    elseif status == DUAL_INFEASIBLE
        println(
                "The problem is dual infeasible. If a primal feasible solution " *
                "exists, the problem is unbounded. To check, set the objective " *
                "to `@objective(model, Min, 0)` and re-solve. If the problem is " *
                "feasible, the primal is unbounded. If the problem is " *
                "infeasible, both the primal and dual are infeasible.",
               )
    elseif status == INFEASIBLE_OR_UNBOUNDED
        println(
                "The model is either infeasible or unbounded. Set the objective " *
                "to `@objective(model, Min, 0)` and re-solve to disambiguate. If " *
                "the problem was infeasible, it will still be infeasible. If the " *
                "problem was unbounded, it will now have a finite optimal solution.",
               )
    else
        println(
                "The model was not solved correctly. The termination status is $status",
               )
    end
    if primal_status(model) in (FEASIBLE_POINT, NEARLY_FEASIBLE_POINT)
        println("  objective value = ", objective_value(model))
    end
    return
end

