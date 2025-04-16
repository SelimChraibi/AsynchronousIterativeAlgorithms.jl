
using Distributed
using DistributedObjects
using LinearAlgebra

"""
    start(algorithm, problem_constructor, stopat; kwargs...)
    start(algorithm, distributed_problem, stopat; kwargs...)

Solve the distributed problem returned by `problem_constructor` (or referenced by `distributed_problem`) using the `algorithm` until the `stopat` conditions are reached.

# Arguments
- `algorithm::AbstractAlgorithm{Q,A}`: subtypes [`AbstractAlgorithm{Q,A}`](@ref) and implementing its functor calls
- `problem_constructor::Function`: for each pid in {`pids` ⋃ current pid}, process pid calling `problem_constructor(pid)` should return the process' assigned problem
- `distributed_problem::DistributedObject`: for each pid in {`pids` ⋃ current pid}, `distributed_problem` should reference process pid's assigned problem on pid
- `stopat::NamedTuple`: you can specify any of the following
    - `iteration::Int64`: maximum number of iterations
    - `epoch::Int64`: maximum number of epochs (an epoch passes when all workers have answered at least one time) 
    - `time::Float64`: maximum elapsed time (in seconds) 
    - other custom stopping conditions that you have specified by implementing [`stopnow`](@ref)

# Keywords
- `saveat=NamedTuple()`: when to record query iterates (`::Q`), iterations, epochs, timestamps (and other custom values specified by implementing [`progress`](@ref)). Specified with any of the following 
    - `iteration::Int64`: save every `iteration`> 0
    - `epoch::Int64`: , save every `epoch`> 0
    - other custom saving conditions that you have specified by implementing [`savenow`](@ref)
- `save_answers=false`: answer iterates (`::A`) along with the pids of the workers that sent them are recorder 
- `pids=workers()`: `pids` of the active workers, you can start a non-distributed (and necessarily synchronous) version of your algorithm with `pids=[1]`
- `synchronous=false`: if `synchronous=true`, the central node waits for all workers to answer before making a step
- `resilience=0`: number of workers allowed to fail before the execution is stopped
- `verbose=1`: if `> 0`, a progress bar is displayed (implent [`progress`](@ref) and/or [`showvalues`](@ref) to customize the display)

# Returns
- NamedTuple: a record of the `queries` and the `iterations`, `epochs`, `timestamps` at which they were recorded, as well an `answer_count` of each worker, additionally, 
    - if `save_answers=true`, a record of the `answers` and the `answers_origin`
    - other custom values you have specified by implementing [`report`](@ref)

# Throws
- `ArgumentError`: if the arguments don't match the specifications.
"""
function start(algorithm::AbstractAlgorithm{Q,A}, problem_constructor::Function, stopat::NamedTuple; saveat=NamedTuple(), save_answers=false, pids=workers(), synchronous=false, resilience=0, verbose=1) where {Q,A}
    check_arguments(algorithm, stopat, saveat, pids, synchronous, resilience)
    recorded_algorithm = RecordedAlgorithm(algorithm, stopat, saveat, pids, save_answers, verbose)
    recorded_start(recorded_algorithm, problem_constructor, pids, synchronous, resilience)
end


function start(algorithm::AbstractAlgorithm{Q,A}, distributed_problem::DistributedObject{M}, stopat::NamedTuple; saveat=NamedTuple(), save_answers=false, pids=workers(), synchronous=false, resilience=0, verbose=1) where {Q,A,M}
    start(algorithm, (pid) -> distributed_problem[], stopat; saveat=saveat, save_answers=save_answers, pids=pids, synchronous=synchronous, resilience=resilience, verbose=verbose)
end

# Start with a RecordedAlgorithm object
function recorded_start(recorded_algorithm::RecordedAlgorithm{Q,A,Alg}, problem_constructor::Function, pids::Vector{Int64}, synchronous::Bool, resilience::Int64) where {Q,A,Alg}
    if pids == [myid()] # non-distributed case
        start_central(recorded_algorithm, problem_constructor)
    else
        open_network(Q, A; pids=pids, resilience=resilience) do network
            for pid in pids
                @async remotecall_fetch(start_worker, pid, network, recorded_algorithm, problem_constructor)
            end
            start_central(network, recorded_algorithm, problem_constructor, synchronous)
        end
    end
    report(recorded_algorithm)
end

# Start the non-distributed loop
function start_central(recorded_algorithm::RecordedAlgorithm{Q,A,Alg}, problem_constructor::Function) where {Q,A,Alg}
    problem = problem_constructor(myid())
    q = recorded_algorithm(problem)

    while !stopnow(recorded_algorithm)
        a = recorded_algorithm(q, problem)
        q = recorded_algorithm(a, 1, problem)
    end
end

#Start a worker's loop
function start_worker(network::Network{Q,A}, recorded_algorithm::RecordedAlgorithm{Q,A,Alg}, problem_constructor::Function) where {Q,A,Alg}
    problem = problem_constructor(myid())
    bound_to(network) do network
        while true
            q = get_query(network)
            a = recorded_algorithm(q, problem)
            send_answer(network, a)
        end
    end
    nothing
end

# Start the central node's loop (asynchronous or synchronous)
function start_central(network::Network{Q,A}, recorded_algorithm::RecordedAlgorithm{Q,A,Alg}, problem_constructor::Function, synchronous::Bool) where {Q,A,Alg}
    problem = problem_constructor(myid())
    q = recorded_algorithm(problem)
    send_query(network, q)

    if !synchronous
        while !stopnow(recorded_algorithm)
            a, worker = get_answer(network)
            q = recorded_algorithm(a, worker, problem)
            send_query(network, q, worker)
        end
    else
        npid = length(network.pids)
        while !stopnow(recorded_algorithm)
            as, workers = Vector{A}(undef, npid), Vector{Int64}(undef, npid)
            for i in 1:npid
                as[i], workers[i] = get_answer(network)
            end
            q = recorded_algorithm(as, workers, problem)
            send_query(network, q)
        end
    end
    nothing
end

function check_arguments(algorithm::AbstractAlgorithm{Q,A}, stopat::NamedTuple, saveat::NamedTuple, pids::Vector{Int64}, synchronous::Bool, resilience::Int64) where {Q,A}
    stopat == NamedTuple() &&
    throw(ArgumentError("You should have at least one valid stopping criterion (try `iteration`, `epoch`, `time`, or `precision`)"))

    myid() ∈ pids && pids ≠ [myid()] && 
    throw(ArgumentError("Current process `$(myid())` cannot be a worker"))

    length(pids) ≤ resilience && 
    throw(ArgumentError("resilience should be strictly smaller than the number of active processes"))

    typeof(algorithm) <: AbstractAlgorithm || 
    throw(ArgumentError("`algorithm` should subtype `AbstractAlgorithm{$Q,$A}`"))

    if synchronous
        haskey(saveat, :iteration) && haskey(saveat, :epoch) && saveat.iteration == saveat.epoch &&
        @warn "`saveat.iteration ≠ saveat.epoch ≠ 0` but epochs and iterations have the same meaning in the synchronous context"
        
        haskey(stopat, :iteration) && haskey(stopat, :epoch) && stopat.iteration == stopat.epoch &&
        @warn "`stopat.iteration ≠ stopat.epoch ≠ 0` but epochs and iterations have the same meaning in the synchronous context"
        
        hasmethod(algorithm, (Vector{A}, Vector{Int64}, Any)) || 
        throw(ArgumentError("`algorithm` should be callable with the signature `algorithm(as::Vector{$A}, workers::Vector{Int64}, problem::Any)"))
        
        return_types = Base.return_types(algorithm, (Vector{A}, Vector{Int64}, Any))
        all([supertype(typeof(algorithm)).parameters[1]] .<: return_types) || 
        throw(ArgumentError("`algorithm(a::Vector{$A}, worker::Vector{Int64}, problem::Any)` should output $Q but its output is or subtypes `$(return_types...)`"))
    else
        pids == [myid()] && 
        throw(ArgumentError("Non-distributed starts (`pids == [myid()]`) are necessarily synchronous"))
        
        hasmethod(algorithm, (A, Int64, Any)) || 
        throw(ArgumentError("`algorithm` should be callable with the signature `algorithm(a::$A, worker::Int64, problem::Any)`"))
    
        return_types = Base.return_types(algorithm, (A, Int64, Any))
        all([supertype(typeof(algorithm)).parameters[1]] .<: return_types) || 
        throw(ArgumentError("`algorithm(a::$A, worker::Int64, problem::Any)` should output $Q but its output is or subtypes `$(return_types...)`"))
    end

    hasmethod(algorithm, (Any,)) || 
    throw(ArgumentError("`algorithm` should be callable with the signature `algorithm(problem::Any)`"))
    
    hasmethod(algorithm, (Q, Any)) || 
    throw(ArgumentError("`algorithm` should be callable with the signature `algorithm(q::$Q, problem::Any)`"))
    
    return_types = Base.return_types(algorithm, (Q, Any))
    all([supertype(typeof(algorithm)).parameters[2]] .<: return_types) || 
    throw(ArgumentError("`algorithm(q::$Q, problem::Any)` should output $A but its output is or subtypes `$(return_types...)`"))
end