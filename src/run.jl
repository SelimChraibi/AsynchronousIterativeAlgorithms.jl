export run

using Distributed
using DistributedObjects
using LinearAlgebra

"""
    run(algorithm::AbstractAlgorithm{Q,A}, problem_constructor::Function, stopat::Union{Tuple{Int64, Int64, Float64}, Tuple{Int64, Int64, Float64, Float64}}; saveat=(0,0), save_answers=false, pids=workers(), synchronous=false, distance::Function=(x::Q,y::Q)->norm(x-y), verbose=1) where {Q,A}

Solve the distributed problem returned by `problem_constructor` using the `algorithm`.

# Arguments
- `algorithm::AbstractAlgorithm{Q,A}`: should be callable with the following signatures
    - `(algorithm::AbstractAlgorithm{Q,A})(problem::Any) where {Q,A}` the initialization step that create the first query iterate `q::Q` 
    - `(algorithm::AbstractAlgorithm{Q,A})(q::Q, problem::Any) where {Q,A}` is the step perfromed by the wokers when they receive a query `q::Q` from the central node
    - `(algorithm::AbstractAlgorithm{Q,A})(a::A, worker::Int64, problem::Any) where {Q,A}` is the step performed by the central node when receiving an answer `a::A` from a worker
    - if `synchronous=true`, `(algorithm::AbstractAlgorithm{Q,A})(as::Vector{A}, workers::Vector{Int64}, problem::Any) where {Q,A}` is the step performed by the central node when it receives the answers `as::Vector{A}` from all the `workers::Vector{Int64}`
    A `algorithm` can additionally define:
    -`AsynchronousOptimizer.stopnow(algorithm::MyAlgorithm)` (with the trait `AsynchronousOptimizer.Stoppability(::MyAlgorithm) = Stoppable()`) to add a stopping condition to `run`'s (iterations, epochs, time) stopping condition
- `problem_constructor::Function`: this function should return process pid's problem when it calls `problem_constructor(pid::Int64)` (for any remote `pids` and on the current pid)
- `stopat::Union{Tuple{Int64, Int64, Float64}, Tuple{Int64, Int64, Float64, Float64}}`: `(i, e, t)` or `(i, e, t, p)` 
    - `i`: maximum number of iterations
    - `e`: maximum number of epochs (all workers have answered at least `e` times) 
    - `t`: maximum runtime `t` (in seconds) 
    - `p`: required precision (in terms of  `distance` between the last two queries)

# Keywords
- `saveat=(0,0)::Tuple{Int64, Int64}`: query iterates (`::Q`) sent by the central nodes are recorded every `i > 0` iterations, `e > 0` epochs in `saveat=(i, e)`
- `save_answers=false::Bool`: answer iterates (`::A`) sent by the workers are recorded
- `pids::Vector{Int64}=workers()`: `pids` of the active workers, you can run a non-distributed (and necessarily synchronous) version of your algorithm with `pids=[1]`
- `synchronous=false`: if `synchronous=true`, the central node waits for all workers to answers before making a step
- `distance::Function=(x::Q,y::Q)->norm(x-y)`: function used to compute the distance between the last two queries
- `verbose=1`: if `> 0`, a progress bar is displayed

# Returns
- NamedTuple: record of the `queries` and the `iterations`, `epochs`, `timestamps` at which they were recorded, as well as `answer_count` of each worker (if `save_answers` is `true`, the `answers` will be recorded with their worker provenance in `answer_origin`)

# Throws
- `ArgumentError`: if the arguments don't match the specifications.
"""
function Base.run(algorithm::AbstractAlgorithm{Q,A}, problem_constructor::Function, stopat::Union{Tuple{Int64, Int64, Float64}, Tuple{Int64, Int64, Float64, Float64}}; saveat=(0,0), save_answers=false, pids=workers(), synchronous=false, distance::Function=(x::Q,y::Q)->norm(x-y), verbose=1) where {Q,A}
    check_arguments(algorithm, stopat, saveat, pids, synchronous)
    algorithm = RecordedAlgorithm{Q,A}(algorithm, true, stopat, saveat, pids, save_answers, distance, verbose)
    start(algorithm, problem_constructor, pids, synchronous)
end


"""
    run(algorithm::AbstractAlgorithm{Q,A}, distributed_problem::DistributedObject{M}, stopat::Union{Tuple{Int64, Int64, Float64}, Tuple{Int64, Int64, Float64, Float64}}; saveat=(0,0), save_answers=false, pids=workers(), synchronous=false, distance::Function=(x::Q,y::Q)->norm(x-y), verbose=1) where {Q,A,M}

Solve the `distributed_problem` using the `algorithm`. Similar to the original `run` function but instead of a `problem_constructor::Function`, a `distributed_problem::DistributedObject` should be passed. `distributed_problem` should reference a problem on the remote `pids` and on the current pid.
"""

function Base.run(algorithm::AbstractAlgorithm{Q,A}, distributed_problem::DistributedObject{M}, stopat::Union{Tuple{Int64, Int64, Float64}, Tuple{Int64, Int64, Float64, Float64}}; saveat=(0,0), save_answers=false, pids=workers(), distance::Function=(x::Q,y::Q)->norm(x-y), synchronous=false, verbose=1) where {Q,A,M}
    run(algorithm, (pid)->distributed_problem[], stopat; saveat=saveat, save_answers=save_answers, pids=pids, synchronous=synchronous, distance=distance, verbose=verbose)
end

"""
    run!(algorithm::AbstractAlgorithm{Q,A}, problem_constructor::Function, stopat::Union{Tuple{Int64, Int64, Float64}, Tuple{Int64, Int64, Float64, Float64}}; saveat=(0,0), save_answers=false, pids=workers(), synchronous=false, distance::Function=(x::Q,y::Q)->norm(x-y), verbose=1) where {Q,A}

Same as `run` but here `algorithm` isn't copied and therefore the original object is modified. This can be useful if you want to keep track of what happens durring the optimization.
"""
function run!(algorithm::AbstractAlgorithm{Q,A}, problem_constructor::Function, stopat::Union{Tuple{Int64, Int64, Float64}, Tuple{Int64, Int64, Float64, Float64}}; saveat=(0,0), save_answers=false, pids=workers(), synchronous=false, distance::Function=(x::Q,y::Q)->norm(x-y), verbose=1) where {Q,A}
    check_arguments(algorithm, stopat, saveat, pids, synchronous)
    algorithm = RecordedAlgorithm{Q,A}(algorithm,false, stopat, saveat, pids, save_answers, distance, verbose)
    start(algorithm, problem_constructor, pids, synchronous)
end

"""
    run!(algorithm::AbstractAlgorithm{Q,A}, distributed_problem::DistributedObject{M}, stopat::Union{Tuple{Int64, Int64, Float64}, Tuple{Int64, Int64, Float64, Float64}}; saveat=(0,0), save_answers=false, pids=workers(), distance::Function=(x::Q,y::Q)->norm(x-y), synchronous=false, verbose=1) where {Q,A,M}
 
Same as `run` but here `algorithm` isn't copied and therefore the original object is modified. This can be useful if you want to keep track of what happens durring the optimization.
"""
function run!(algorithm::AbstractAlgorithm{Q,A}, distributed_problem::DistributedObject{M}, stopat::Union{Tuple{Int64, Int64, Float64}, Tuple{Int64, Int64, Float64, Float64}}; saveat=(0,0), save_answers=false, pids=workers(), distance::Function=(x::Q,y::Q)->norm(x-y), synchronous=false, verbose=1) where {Q,A,M}
    run!(algorithm, (pid)->distributed_problem[], stopat; saveat=saveat, save_answers=save_answers, pids=pids, synchronous=synchronous, distance=distance, verbose=verbose)
end

function start(algorithm::RecordedAlgorithm{Q,A}, problem_constructor::Function, pids::Vector{Int64}, synchronous::Bool) where {Q,A}
    if pids==[myid()] # non-distributed case
        start_central(algorithm, problem_constructor)
    else
        Network{Q,A}(pids) do network
            for pid in pids
                remotecall_fetch(start_worker, pid, network, algorithm, problem_constructor)
            end
            start_central(network, algorithm, problem_constructor, synchronous)
        end
    end
    report(algorithm)
end

"""
function start_central(algorithm::RecordedAlgorithm{Q,A}, problem_constructor::Function) where {Q,A}

Start the non-distributed loop
"""
function start_central(algorithm::RecordedAlgorithm{Q,A}, problem_constructor::Function) where {Q,A}
    problem = problem_constructor(myid())
    q = algorithm(problem)
    
    while !stopnow(algorithm)
        a = algorithm(q, problem)
        q = algorithm(a, 1, problem)
    end
end

"""
    start_worker(network::Network{Q,A}, algorithm::AbstractAlgorithm{Q,A}, problem_constructor::Function) where {Q,A,M}

Start a worker's loop
"""
function start_worker(network::Network{Q,A}, algorithm::AbstractAlgorithm{Q,A}, problem_constructor::Function) where {Q,A}
    problem = problem_constructor(myid())
    task = @task while isopen(network) 
        q = get_query(network)
        a = algorithm(q, problem)
        send_answer(network, a)
    end
    task = bind(network, task)
    schedule(errormonitor(task))
end

"""
function start_central(network::Network{Q,A}, algorithm::RecordedAlgorithm{Q,A}, problem_constructor::Function, synchronous::Bool) where {Q,A} 

Start the central node's loop (asynchronous or synchronous)
"""
function start_central(network::Network{Q,A}, algorithm::RecordedAlgorithm{Q,A}, problem_constructor::Function, synchronous::Bool) where {Q,A}
    problem = problem_constructor(myid())
    q = algorithm(problem)
    send_query(network, q)

    if !synchronous
        while !stopnow(algorithm)
            a, worker = get_answer(network)
            q = algorithm(a, worker, problem)
            send_query(network, q, worker)
        end
    else 
        npid = length(network.pids)
        while !stopnow(algorithm)
            as, workers = Vector{A}(undef, npid), Vector{Int64}(undef, npid)
            for i in 1:npid
                as[i], workers[i] = get_answer(network)
            end
            q = algorithm(as, workers, problem)
            send_query(network, q)
        end
    end
end

function check_arguments(algorithm::AbstractAlgorithm{Q,A}, stopat::Union{Tuple{Int64, Int64, Float64}, Tuple{Int64, Int64, Float64, Float64}}, saveat::Tuple{Int64, Int64}, pids::Vector{Int64}, synchronous::Bool) where {Q,A} 
    stopat == (0,0,0.) && Stoppability(algorithm)==NotStoppable() && throw(Core.ArgumentError("You should have a stopping criterion"))
    myid() ∈ pids && pids ≠ [myid()] && throw(Core.ArgumentError("Current process \"$(myid())\" cannot be a worker"))

    typeof(algorithm) <: AbstractAlgorithm || throw(Core.ArgumentError("\"algorithm\" should subtype \"AbstractAlgorithm{$Q,$A}\""))

    if synchronous
        prod(saveat) ≠ 0 && diff(saveat) ≠ 0 && @warn "\"saveat[1] ≠ saveat[2] ≠ 0\" but epochs and iterations have the same meaning in the synchronous context"
        prod(stopat[1:2]) ≠ 0 && diff(stopat[1:2]) ≠ 0 && @warn "\"stopat[1] ≠ stopat[2] ≠ 0\" but epochs and iterations have the same meaning in the synchronous context"
        hasmethod(algorithm, (Vector{A}, Vector{Int64}, Any)) || throw(Core.ArgumentError("\"algorithm\" should be callable with the signature \"algorithm(as::Vector{$A}, workers::Vector{Int64}, problem::Any)"))
        return_types = Base.return_types(algorithm, (Vector{A}, Vector{Int64}, Any))
        
        all([supertype(typeof(algorithm)).parameters[1]] .<: return_types) || throw(Core.ArgumentError("\"algorithm(a::Vector{$A}, worker::Vector{Int64}, problem::Any)\" should output $Q but its output is or subtypes \"$(return_types...)\""))
    else
        pids == [myid()] && throw(Core.ArgumentError("Non-distributed runs (\"pids == [myid()]\") are necessarily synchronous"))
        hasmethod(algorithm, (A, Int64, Any)) || throw(Core.ArgumentError("\"algorithm\" should be callable with the signature \"algorithm(a::$A, worker::Int64, problem::Any)\""))
        return_types = Base.return_types(algorithm, (A, Int64, Any))
        all([supertype(typeof(algorithm)).parameters[1]] .<: return_types) || throw(Core.ArgumentError("\"algorithm(a::$A, worker::Int64, problem::Any)\" should output $Q but its output is or subtypes \"$(return_types...)\""))
    end

    hasmethod(algorithm, (Any,)) || throw(Core.ArgumentError("\"algorithm\" should be callable with the signature \"algorithm(problem::Any)\""))
    hasmethod(algorithm, (Q, Any)) || throw(Core.ArgumentError("\"algorithm\" should be callable with the signature \"algorithm(q::$Q, problem::Any)\""))
    return_types = Base.return_types(algorithm, (Q, Any))
    all([supertype(typeof(algorithm)).parameters[2]] .<: return_types) || throw(Core.ArgumentError("\"algorithm(q::$Q, problem::Any)\" should output $A but its output is or subtypes \"$(return_types...)\""))
end 