export AggregationAlgorithm, AveragingAlgorithm

initialize(algorithm::AbstractAlgorithm, problem::Any) = throw(ArgumentError("Method initialize(::$(typeof(algorithm)), problem::Any) not implemented."))
aggregate(algorithm::AbstractAlgorithm{Q,A}, as::Vector{A}, workers::Vector{Int64}) where {Q,A} = throw(ArgumentError("Method aggregate(::$(typeof(algorithm)), problem::Any) not implemented."))
query(algorithm::AbstractAlgorithm, agg::AggregatedA, problem::Any) where {AggregatedA} = throw(ArgumentError("Method query(::$(typeof(algorithm)), problem::Any) not implemented."))
answer(algorithm::AbstractAlgorithm{Q,A}, q::Q, problem::Any) where {Q,A} = throw(ArgumentError("Method answer(::$(typeof(algorithm)), problem::Any) not implemented."))

"""
    AggregationAlgorithm(arg; kwarg)::AbstractAlgorithm

Distributed algorithm that writes: `q_j <- query(aggregate([answer(q_i) for i in connected]))`
Where a "connected" worker is a worker that has answered at least once.
(Not memory optimized: `length(pids)` answers are stored on the central worker at all times)

# Argument
- `algorithm<:AbstractAlgorithm{Q,A}` which should define the following (where `const AIA = AsynchronousIterativeAlgorithms`)
    - `AIA.initialize(algorithm, problem::Any)::Q`: step that creates the first query iterate
    - `AIA.aggregate(algorithm, as::Vector{A}, workers::Vector{Int64})::AggregatedA` where A: step performed by the central node when receiving the answers `as::Vector{A}` from the `workers`
    - `AIA.query(algorithm, agg::AggregatedA, problem::Any)::Q`: step producing a query from the aggregated answer `agg::AggregatedA`, performed by the central node
    - `AIA.answer(algorithm, q::Q, problem::Any)::A`: step perfromed by the wokers when they receive a query `q::Q` from the central node

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
    (::AggregationAlgorithm{Q,A,Alg})(problem::Any)::Q where Alg<:AbstractAlgorithm{Q,A} where {Q,A}

The initialization step that create the first query iterate
"""
function (agg::AggregationAlgorithm)(problem::Any)
    initialize(agg.algorithm,  problem)
end

"""
    (::AggregationAlgorithm{Q,A,Alg})(a::A, worker::Int64, problem::Any)::Q where Alg<:AbstractAlgorithm{Q,A} where {Q,A}

Asynchronous step performed by the central node when receiving an answer `a::A` from a worker
"""
function (agg::AggregationAlgorithm{Q,A,Alg})(a::A, worker::Int64, problem::Any) where Alg<:AbstractAlgorithm{Q,A} where {Q,A}
    agg.connected[worker] = true
    agg.answers[worker] = a
    query(agg.algorithm, aggregate(agg.algorithm, agg.answers[agg.connected], (1:maximum(agg.pids))[agg.connected]), problem)
end

"""
    (::AggregationAlgorithm{Q,A,Alg})(as::Vector{A}, workers::Vector{Int64}, problem::Any)::Q where Alg<:AbstractAlgorithm{Q,A} where {Q,A}

Synchronous step performed by the central node when receiving answers `as::Vector{A}` respectively from `workers::Vector{Int64}`
"""
function (agg::AggregationAlgorithm{Q,A,Alg})(as::Vector{A}, workers::Vector{Int64}, problem::Any) where Alg<:AbstractAlgorithm{Q,A} where {Q,A}
    query(agg.algorithm, aggregate(agg.algorithm,  as, workers), problem)
end

"""
    (::AggregationAlgorithm{Q,A,Alg})(q::Q, problem::Any)->A where Alg<:AbstractAlgorithm{Q,A} where {Q,A}

Steps performed by the workers when they receive a query `q::Q` from the central node
"""
function (agg::AggregationAlgorithm{Q,A,Alg})(q::Q, problem::Any) where Alg<:AbstractAlgorithm{Q,A} where {Q,A}
    answer(agg.algorithm,  q, problem)
end

stopnow(agg::AggregationAlgorithm, stopat::NamedTuple) = stopnow(agg.algorithm, stopat)
showvalues(agg::AggregationAlgorithm) = showvalues(agg.algorithm)
report(agg::AggregationAlgorithm) = report(agg.algorithm)
progress(agg::AggregationAlgorithm, stopat::NamedTuple) = progress(agg.algorithm, stopat)
savenow(agg::AggregationAlgorithm, saveat::NamedTuple) = savenow(agg.algorithm, saveat) 
savevalues(agg::AggregationAlgorithm) = savevalues(agg.algorithm)


"""
    AveragingAlgorithm(arg; kwarg)::AbstractAlgorithm

Distributed algorithm that writes: `q_j <- query(weighted_average([answer(q_i) for i in connected]))`
Where a "connected" worker is a worker that has answered at least once.
(Memory optimized: only the equivalent of one answer is stored on the central worker at all times)

# Argument
- `algorithm<:AbstractAlgorithm{Q,A}` which should define the following (where `const AIA = AsynchronousIterativeAlgorithms`)
    - `AIA.initialize(algorithm, problem::Any)::Q`: step that creates the first query iterate
    - `AIA.query(algorithm, a::A, problem::Any)::Q`: step producing a query from the averaged answer, performed by the central node
    - `AIA.answer(algorithm, q::Q, problem::Any)::A`: step perfromed by the wokers when they receive a query `q::Q` from the central node

# Keyword
- `pids=workers()`: `pids` of the active workers
- `weights=ones(length(pids))`: weights of each pid in the weighted average
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
    function AveragingAlgorithm(algorithm::Alg; pids=procs(), weights=ones(length(pids))) where Alg<:AbstractAlgorithm{Q,A} where {Q,A}
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
    (::AveragingAlgorithm{Q,A,Alg})(problem::Any) where Alg<:AbstractAlgorithm{Q,A} where {Q,A}

The initialization step that create the first query iterate
"""
function (avg::AveragingAlgorithm)(problem::Any)
    initialize(avg.algorithm, problem)
end

"""
    (::AveragingAlgorithm{Q,A,Alg})(δas::Vector{A}, workers::Vector{Int64}, problem::Any) where Alg<:AbstractAlgorithm{Q,A} where {Q,A}

Asynchronous step performed by the central node when receiving an answer `a::A` from a worker.
"""
function (avg::AveragingAlgorithm{Q,A,Alg})(δa::A, worker::Int64, problem::Any) where Alg<:AbstractAlgorithm{Q,A} where {Q,A}
    avg.connected[worker] = true
    normalization = sum(avg.connected .* avg.weights)
    avg.last_average = isnothing(avg.last_average) ? δa : (avg.weights[worker] * δa + avg.last_normalization * avg.last_average) / normalization
    avg.last_normalization = normalization
    query(avg.algorithm, avg.last_average, problem)
end

"""
    (::AveragingAlgorithm{Q,A,Alg})(δas::Vector{A}, workers::Vector{Int64}, problem::Any) where Alg<:AbstractAlgorithm{Q,A} where {Q,A}

Synchronous step performed by the central node when receiving answers `a::Vector{A}` respectively from `workers::Vector{Int64}`
"""
function (avg::AveragingAlgorithm{Q,A,Alg})(δas::Vector{A}, workers::Vector{Int64}, problem::Any) where Alg<:AbstractAlgorithm{Q,A} where {Q,A}
    avg.last_average = isnothing(avg.last_average) ? δas : sum(avg.weights[workers] * δas) / sum(avg.weights) + avg.last_average
    query(avg.algorithm,  avg.last_average, problem)
end

"""
 (::AveragingAlgorithm{Q,A,Alg})(q::Q, problem::Any) where Alg<:AbstractAlgorithm{Q,A} where {Q,A}

Steps performed by the workers when they receive a query `q::Q` from the central node
"""
function (avg::AveragingAlgorithm{Q,A,Alg})(q::Q, problem::Any) where Alg<:AbstractAlgorithm{Q,A} where {Q,A}
    a = answer(avg.algorithm,  q, problem)
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