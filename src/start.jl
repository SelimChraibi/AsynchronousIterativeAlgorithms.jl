export start, start!

using Distributed
using DistributedObjects
using LinearAlgebra

"""
    start(algorithm::AbstractAlgorithm{Q,A}, problem_constructor::Function, stopat::Union{Tuple{Int64, Int64, Float64}, Tuple{Int64, Int64, Float64, Float64}}; saveat=(0,0), save_answers=false, pids=workers(), synchronous=false, distance::Function=(x::Q,y::Q)->norm(x-y), resilience=0, verbose=1) where {Q,A}

Solve the distributed problem returned by `problem_constructor` using the `algorithm`.

# Arguments
- `algorithm::AbstractAlgorithm{Q,A}`: subtyping [`AbstractAlgorithm{Q,A}`](@ref) and implementing its functor calls
- `problem_constructor::Function`: this function should return the process pid's problem when it calls `problem_constructor(pid::Int64)` (for any remote `pids` and on the current pid)
- `stopat::Union{Tuple{Int64, Int64, Float64}, Tuple{Int64, Int64, Float64, Float64}}`: `(i, e, t)` or `(i, e, t, p)` 
    - `i`: maximum number of iterations
    - `e`: maximum number of epochs (all workers have answered at least `e` times) 
    - `t`: maximum starttime `t` (in seconds) 
    - `p`: required precision (in terms of `distance` between the last two queries)

# Keywords
- `saveat=(0,0)::Tuple{Int64, Int64}`: query iterates (`::Q`) sent by the central nodes are recorded every `i > 0` iterations, `e > 0` epochs in `saveat=(i, e)`
- `save_answers=false::Bool`: answer iterates (`::A`) sent by the workers are recorded
- `pids::Vector{Int64}=workers()`: `pids` of the active workers, you can start a non-distributed (and necessarily synchronous) version of your algorithm with `pids=[1]`
- `synchronous=false`: if `synchronous=true`, the central node waits for all workers to answer before making a step
- `distance::Function=(x::Q,y::Q)->norm(x-y)`: function used to compute the distance between the last two queries
- `resilience::Int64=0`: number of workers allowed to fail before the execution is stopped
- `verbose=1`: if `> 0`, a progress bar is displayed

# Returns
- NamedTuple: a record of the `queries` and the `iterations`, `epochs`, `timestamps` at which they were recorded, as well as `answer_count` of each worker (if `save_answers` is `true`, the `answers` will be recorded with their worker provenance in `answer_origin`)

# Throws
- `ArgumentError`: if the arguments don't match the specifications.

    start(algorithm::AbstractAlgorithm{Q,A}, distributed_problem::DistributedObject{M}, stopat::Union{Tuple{Int64, Int64, Float64}, Tuple{Int64, Int64, Float64, Float64}}; saveat=(0,0), save_answers=false, pids=workers(), synchronous=false, distance::Function=(x::Q,y::Q)->norm(x-y), resilience=0, verbose=1) where {Q,A,M}

Solve the `distributed_problem` using the `algorithm`. Similar to the original `start` function but instead of a `problem_constructor::Function`, a `distributed_problem::DistributedObject` should be passed. `distributed_problem` should reference a problem on the remote `pids` and on the current pid.
"""
function start(algorithm::AbstractAlgorithm{Q,A}, problem_constructor::Function, stopat::Union{Tuple{Int64,Int64,Float64},Tuple{Int64,Int64,Float64,Float64}}; saveat=(0, 0), save_answers=false, pids=workers(), synchronous=false, distance::Function=(x::Q, y::Q) -> norm(x - y), resilience=0, verbose=1) where {Q,A}
    check_arguments(algorithm, stopat, saveat, pids, synchronous, resilience)
    recorded_algorithm = RecordedAlgorithm{Q,A}(algorithm, true, stopat, saveat, pids, save_answers, distance, verbose)
    recorded_start(recorded_algorithm, problem_constructor, pids, synchronous, resilience)
end


function start(algorithm::AbstractAlgorithm{Q,A}, distributed_problem::DistributedObject{M}, stopat::Union{Tuple{Int64,Int64,Float64},Tuple{Int64,Int64,Float64,Float64}}; saveat=(0, 0), save_answers=false, pids=workers(), synchronous=false, distance::Function=(x::Q, y::Q) -> norm(x - y), resilience=0, verbose=1) where {Q,A,M}
    start(algorithm, (pid) -> distributed_problem[], stopat; saveat=saveat, save_answers=save_answers, pids=pids, synchronous=synchronous, distance=distance, resilience=resilience, verbose=verbose)
end

"""
    start!(algorithm::AbstractAlgorithm{Q,A}, problem_constructor::Function, stopat::Union{Tuple{Int64, Int64, Float64}, Tuple{Int64, Int64, Float64, Float64}}; saveat=(0,0), save_answers=false, pids=workers(), synchronous=false, distance::Function=(x::Q,y::Q)->norm(x-y), resilience=0, verbose=1) where {Q,A}

Same as [`start`](@ref) but `start!` uses a deep copy of your algorithm and won't modify it. This version enables modifications. This can be useful to record information during the execution for example.

    start!(algorithm::AbstractAlgorithm{Q,A}, distributed_problem::DistributedObject{M}, stopat::Union{Tuple{Int64, Int64, Float64}, Tuple{Int64, Int64, Float64, Float64}}; saveat=(0,0), save_answers=false, pids=workers(), synchronous=false, distance::Function=(x::Q,y::Q)->norm(x-y), resilience=0, verbose=1) where {Q,A,M}

Same as [`start`](@ref) but `start!` uses a deep copy of your algorithm and won't modify it. This version enables modifications. This can be useful to record information during the execution for example.
"""
function start!(algorithm::AbstractAlgorithm{Q,A}, problem_constructor::Function, stopat::Union{Tuple{Int64,Int64,Float64},Tuple{Int64,Int64,Float64,Float64}}; saveat=(0, 0), save_answers=false, pids=workers(), synchronous=false, distance::Function=(x::Q, y::Q) -> norm(x - y), resilience=0, verbose=1) where {Q,A}
    check_arguments(algorithm, stopat, saveat, pids, synchronous, resilience)
    algorithm = RecordedAlgorithm{Q,A}(algorithm, false, stopat, saveat, pids, save_answers, distance, verbose)
    recorded_start(recorded_algorithm, problem_constructor, pids, synchronous, resilience)
end

function start!(algorithm::AbstractAlgorithm{Q,A}, distributed_problem::DistributedObject{M}, stopat::Union{Tuple{Int64,Int64,Float64},Tuple{Int64,Int64,Float64,Float64}}; saveat=(0, 0), save_answers=false, pids=workers(), synchronous=false, distance::Function=(x::Q, y::Q) -> norm(x - y), resilience=0, verbose=1) where {Q,A,M}
    start!(algorithm, (pid) -> distributed_problem[], stopat; saveat=saveat, save_answers=save_answers, pids=pids, synchronous=synchronous, distance=distance, resilience=resilience, verbose=verbose)
end

# Start with a RecordedAlgorithm object
function recorded_start(recorded_algorithm::RecordedAlgorithm{Q,A}, problem_constructor::Function, pids::Vector{Int64}, synchronous::Bool, resilience::Int64) where {Q,A}
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
function start_central(recorded_algorithm::RecordedAlgorithm{Q,A}, problem_constructor::Function) where {Q,A}
    problem = problem_constructor(myid())
    q = recorded_algorithm(problem)

    while !stopnow(recorded_algorithm)
        a = recorded_algorithm(q, problem)
        q = recorded_algorithm(a, 1, problem)
    end
end

#Start a worker's loop
function start_worker(network::Network{Q,A}, recorded_algorithm::RecordedAlgorithm{Q,A}, problem_constructor::Function) where {Q,A}
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
function start_central(network::Network{Q,A}, recorded_algorithm::RecordedAlgorithm{Q,A}, problem_constructor::Function, synchronous::Bool) where {Q,A}
    problem = problem_constructor(myid())
    q = recorded_algorithm(problem)
    send_query(network, q)

    if !synchronous
        while !stopnow(recorded_algorithm) #&& isopen(network) 
            a, worker = get_answer(network)
            q = recorded_algorithm(a, worker, problem)
            send_query(network, q, worker)
        end
    else
        npid = length(network.pids)
        while !stopnow(recorded_algorithm) #&& isopen(network) 
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

function check_arguments(algorithm::AbstractAlgorithm{Q,A}, stopat::Union{Tuple{Int64,Int64,Float64},Tuple{Int64,Int64,Float64,Float64}}, saveat::Tuple{Int64,Int64}, pids::Vector{Int64}, synchronous::Bool, resilience::Int64) where {Q,A}
    stopat == (0, 0, 0.0) && Stoppability(algorithm) == NotStoppable() && throw(Core.ArgumentError("You should have a stopping criterion"))
    myid() ∈ pids && pids ≠ [myid()] && throw(Core.ArgumentError("Current process \"$(myid())\" cannot be a worker"))
    length(pids) ≤ resilience && throw(Core.ArgumentError("resilience should be strictly smaller than the number of active processes"))

    typeof(algorithm) <: AbstractAlgorithm || throw(Core.ArgumentError("\"algorithm\" should subtype \"AbstractAlgorithm{$Q,$A}\""))

    if synchronous
        prod(saveat) ≠ 0 && diff(saveat) ≠ 0 && @warn "\"saveat[1] ≠ saveat[2] ≠ 0\" but epochs and iterations have the same meaning in the synchronous context"
        prod(stopat[1:2]) ≠ 0 && diff(stopat[1:2]) ≠ 0 && @warn "\"stopat[1] ≠ stopat[2] ≠ 0\" but epochs and iterations have the same meaning in the synchronous context"
        hasmethod(algorithm, (Vector{A}, Vector{Int64}, Any)) || throw(Core.ArgumentError("\"algorithm\" should be callable with the signature \"algorithm(as::Vector{$A}, workers::Vector{Int64}, problem::Any)"))
        return_types = Base.return_types(algorithm, (Vector{A}, Vector{Int64}, Any))

        all([supertype(typeof(algorithm)).parameters[1]] .<: return_types) || throw(Core.ArgumentError("\"algorithm(a::Vector{$A}, worker::Vector{Int64}, problem::Any)\" should output $Q but its output is or subtypes \"$(return_types...)\""))
    else
        pids == [myid()] && throw(Core.ArgumentError("Non-distributed starts (\"pids == [myid()]\") are necessarily synchronous"))
        hasmethod(algorithm, (A, Int64, Any)) || throw(Core.ArgumentError("\"algorithm\" should be callable with the signature \"algorithm(a::$A, worker::Int64, problem::Any)\""))
        return_types = Base.return_types(algorithm, (A, Int64, Any))
        all([supertype(typeof(algorithm)).parameters[1]] .<: return_types) || throw(Core.ArgumentError("\"algorithm(a::$A, worker::Int64, problem::Any)\" should output $Q but its output is or subtypes \"$(return_types...)\""))
    end

    hasmethod(algorithm, (Any,)) || throw(Core.ArgumentError("\"algorithm\" should be callable with the signature \"algorithm(problem::Any)\""))
    hasmethod(algorithm, (Q, Any)) || throw(Core.ArgumentError("\"algorithm\" should be callable with the signature \"algorithm(q::$Q, problem::Any)\""))
    return_types = Base.return_types(algorithm, (Q, Any))
    all([supertype(typeof(algorithm)).parameters[2]] .<: return_types) || throw(Core.ArgumentError("\"algorithm(q::$Q, problem::Any)\" should output $A but its output is or subtypes \"$(return_types...)\""))
end