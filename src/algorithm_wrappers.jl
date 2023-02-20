export AggregationAlgorithm, AveragingAlgorithm

"""
    AggregationAlgorithm{Q,A,Alg<:AbstractAlgorithm{Q,A}}(arg; kwarg)::AbstractAlgorithm{Q,A} where {Q,A}

Distributed algorithm that writes: `q_j <- query(aggregate([answer(q_i) for i in connected]))`
Where a "connected" worker is a worker that has answered at least once.
(Not memory optimized: `length(pids)` answers are stored on the central worker at all times)

# Argument
- `algorithm<:AbstractAlgorithm{Q,A}` which should define the following
    - `algorithm(problem::Any)::Q`: the initialization step that create the first query iterate
    - `algorithm(as::Vector{A}, workers::Vector{Int64})::AggregatedA` where A: the answer aggregarion step performed by the central node when receiving the answers `as::Vector{A}` from the `workers`
    - `algorithm(agg::AggregatedA, problem::Any)::Q`: the query step producing a query from the aggregated answer `agg::AggregatedA`, performed by the central node
    - `algorithm(q::Q, problem::Any)::A`: the answer step perfromed by the wokers when they receive a query `q::Q` from the central node

# Keyword
- `pids=workers()`: `pids` of the active workers
"""
struct AggregationAlgorithm{Q,A,Alg<:AbstractAlgorithm{Q,A}} <: AbstractAlgorithm{Q,A}
    algorithm::Alg
    pids::Vector{Int64}
    answers::Vector{A}
    connected::BitVector
    function AggregationAlgorithm(algorithm::Alg; pids=workers()) where Alg<:AbstractAlgorithm{Q,A} where {Q,A}
        connected = BitVector(zeros(maximum(pids)))
        answers = Vector{A}(undef, maximum(pids))
        new{Q,A,Alg}(algorithm, pids, answers, connected)
    end
end

"""
    (::AggregationAlgorithm{Q,A,Alg})(problem::Any)::Q where {Q,A,Alg}

The initialization step that create the first query iterate
"""
function (agg::AggregationAlgorithm)(problem::Any)
    agg.algorithm(problem)
end

"""
    (::AggregationAlgorithm{Q,A,Alg})(a::A, worker::Int64, problem::Any)::Q where {Q,A,Alg}

Asynchronous step performed by the central node when receiving an answer `a::A` from a worker
"""
function (agg::AggregationAlgorithm{Q,A,Alg})(a::A, worker::Int64, problem::Any) where {Q,A,Alg}
    agg.connected[worker] = true
    agg.answers[worker] = a
    agg.algorithm(agg.algorithm(agg.answers[agg.connected], (1:maximum(agg.pids))[agg.connected]), problem)
end

"""
    (::AggregationAlgorithm{Q,A,Alg})(as::Vector{A}, workers::Vector{Int64}, problem::Any)::Q where {Q,A,Alg}

Synchronous step performed by the central node when receiving answers `as::Vector{A}` respectively from `workers::Vector{Int64}`
"""
function (agg::AggregationAlgorithm{Q,A,Alg})(as::Vector{A}, workers::Vector{Int64}, problem::Any) where {Q,A,Alg}
    agg.algorithm(agg.algorithm(as, workers), problem)
end

"""
    (::AggregationAlgorithm{Q,A,Alg})(q::Q, problem::Any)->A where {Q,A,Alg}

Steps performed by the workers when they receive a query `q::Q` from the central node
"""
function (agg::AggregationAlgorithm{Q,A,Alg})(q::Q, problem::Any) where {Q,A,Alg}
    agg.algorithm(q, problem)
end

stopnow(agg::AggregationAlgorithm, stopat::NamedTuple) = stopnow(agg.algorithm, stopat)
showvalues(agg::AggregationAlgorithm) = showvalues(agg.algorithm)
report(agg::AggregationAlgorithm) = report(agg.algorithm)
progress(agg::AggregationAlgorithm, stopat::NamedTuple) = progress(agg.algorithm, stopat)
savenow(agg::AggregationAlgorithm, saveat::NamedTuple) = savenow(agg.algorithm, saveat) 
savevalues(agg::AggregationAlgorithm) = savevalues(agg.algorithm)


"""
    AveragingAlgorithm{Q,A,Alg<:AbstractAlgorithm{Q,A}}(arg; kwarg)::AbstractAlgorithm{Q,A} where {Q,A}

Distributed algorithm that writes: `q_j <- query(weighted_average([answer(q_i) for i in connected]))`
Where a "connected" worker is a worker that has answered at least once.
(Memory optimized: only the equivalent of one answer is stored on the central worker at all times)

# Argument
- `algorithm<:AbstractAlgorithm{Q,A}` which should define the following
    - `algorithm(problem::Any)::Q`: the initialization step that create the first query iterate
    - `algorithm(a::A, problem::Any)::Q`: the query step producing a query from the averaged answer, performed by the central node
    - `algorithm(q::Q, problem::Any)::A`: the answer step perfromed by the wokers when they receive a query `q::Q` from the central node

# Keyword
- `pids=workers()`: `pids` of the active workers
"""
mutable struct AveragingAlgorithm{Q,A,Alg<:AbstractAlgorithm{Q,A}} <: AbstractAlgorithm{Q,A}
    pids::Vector{Int64}
    algorithm::Alg
    connected::BitVector
    last_normalization::Float64
    last_answer::Union{A,Nothing}
    last_answers::Vector{A}
    last_average::Union{A,Nothing}
    weights::Vector{Float64}
    function AveragingAlgorithm(algorithm::Alg; pids=procs(), weights=ones(nprocs())) where Alg<:AbstractAlgorithm{Q,A} where {Q,A}
        @assert length(pids) == length(weights) "There should be as many weights as there are pids"
        maxpid = maximum(pids)
        connected = BitVector(zeros(maxpid))
        sparse_weights = zeros(maxpid)
        for (pid, weight) in zip(pids, weights)
            sparse_weights[pid] = weight
        end
        last_normalization = 1.0
        last_answer = nothing
        last_answers = Vector{A}(undef, maximum(pids))
        last_average = nothing

        new{Q,A,Alg}(pids, algorithm, connected, last_normalization, last_answer, last_answers, last_average, sparse_weights)
    end
end

"""
    (::AveragingAlgorithm{Q,A,Alg})(problem::Any) where {Q,A,Alg}

The initialization step that create the first query iterate
"""
function (avg::AveragingAlgorithm)(problem::Any)
    avg.algorithm(problem)
end

"""
    (::AveragingAlgorithm{Q,A,Alg})(δas::Vector{A}, workers::Vector{Int64}, problem::Any) where {Q,A,Alg}

Asynchronous step performed by the central node when receiving an answer `a::A` from a worker.
"""
function (avg::AveragingAlgorithm{Q,A,Alg})(δa::A, worker::Int64, problem::Any) where {Q,A,Alg}
    avg.connected[worker] = true
    normalization = sum(avg.connected .* avg.weights)
    avg.last_average = isnothing(avg.last_average) ? δa : (avg.weights[worker] * δa + avg.last_normalization * avg.last_average) / normalization
    avg.last_normalization = normalization
    avg.algorithm(avg.last_average, problem)
end

"""
    (::AveragingAlgorithm{Q,A,Alg})(δas::Vector{A}, workers::Vector{Int64}, problem::Any) where {Q,A,Alg}

Synchronous step performed by the central node when receiving answers `a::Vector{A}` respectively from `workers::Vector{Int64}`
"""
function (avg::AveragingAlgorithm{Q,A,Alg})(δas::Vector{A}, workers::Vector{Int64}, problem::Any) where {Q,A,Alg}
    avg.last_average = isnothing(avg.last_average) ? δas : sum(avg.weights[workers] * δas) / sum(avg.weights) + avg.last_average
    avg.algorithm(avg.last_average, problem)
end

"""
 (::AveragingAlgorithm{Q,A,Alg})(q::Q, problem::Any) where {Q,A,Alg}

Steps performed by the workers when they receive a query `q::Q` from the central node
"""
function (avg::AveragingAlgorithm{Q,A,Alg})(q::Q, problem::Any) where {Q,A,Alg}
    a = avg.algorithm(q, problem)
    δa = isnothing(avg.last_answer) ? a : a - avg.last_answer
    avg.last_answer = a
    return δa
end

stopnow(avg::AveragingAlgorithm, stopat::NamedTuple) = stopnow(avg.algorithm, stopat)
showvalues(avg::AveragingAlgorithm) = showvalues(avg.algorithm)
report(avg::AveragingAlgorithm) = report(avg.algorithm)
progress(avg::AveragingAlgorithm, stopat::NamedTuple) = progress(avg.algorithm, stopat)
savenow(avg::AveragingAlgorithm, saveat::NamedTuple) = savenow(avg.algorithm, saveat) 
savevalues(avg::AveragingAlgorithm) = savevalues(avg.algorithm)