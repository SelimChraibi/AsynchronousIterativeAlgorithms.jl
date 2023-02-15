export AggregationAlgorithm, AveragingAlgorithm

"""
    AggregationAlgorithm{Q,A}(args...; kwargs...) -> AggregationAlgorithm{Q,A} <: AbstractAlgorithm{Q,A}

Distributed algorithm that writes: `q_j <- query(aggregate([answer(q_i) for i in connected]))`
Where a "connected" worker is a worker that has answered at least once.
(Not memory optimized: `length(pids)` answers are stored on the central worker at all times)

# Arguments
- `initialize::Function`: with signature `initialize(problem::Any)`
- `aggregate::Function`: with signature `aggregate(a::Vector{A}, workers::Vector{Int64})` where `workers` lists the provenance of the elements of `a` 
- `query::Function`: with signature `query(a::A, problem::Any)`
- `answer::Function`: with signature `answer(q::Q, problem::Any)`

# Keyword
- `pids=workers()`: 
"""
struct AggregationAlgorithm{Q,A} <: AbstractAlgorithm{Q,A}
    pids::Vector{Int64}
    initialize::Function
    aggregate::Function
    query::Function
    answer::Function
    answers::Vector{A}
    connected::BitVector
    function AggregationAlgorithm{Q,A}(initialize::Function, aggregate::Function, query::Function, answer::Function; pids=workers()) where {Q,A}
        connected = BitVector(zeros(maximum(pids)))
        answers = Vector{A}(undef, maximum(pids))
        new(pids, initialize, aggregate, query, answer, answers, connected)
    end
end

"""
    (::AggregationAlgorithm{Q,A})(problem::Any) where {Q,A}

Initialization step: computing the first iterate 
"""
function (agg::AggregationAlgorithm{Q,A})(problem::Any) where {Q,A}
    agg.initialize(problem)
end

"""
    (::AggregationAlgorithm{Q,A})(a::A, worker::Int64, problem::Any) where {Q,A}

Asynchronous step performed by the central node when receiving an answer `a::A` from a worker
"""
function (agg::AggregationAlgorithm{Q,A})(a::A, worker::Int64, problem::Any) where {Q,A}
    agg.connected[worker] = true
    agg.answers[worker] = a
    agg.query(agg.aggregate(agg.answers[agg.connected], (1:maximum(agg.pids))[agg.connected]), problem)
end

"""
    (::AggregationAlgorithm{Q,A})(as::Vector{A}, workers::Vector{Int64}, problem::Any) where {Q,A}

Synchronous step performed by the central node when receiving answers `a::Vector{A}` respectively from `workers::Vector{Int64}`
"""
function (agg::AggregationAlgorithm{Q,A})(as::Vector{A}, workers::Vector{Int64}, problem::Any) where {Q,A}
    agg.query(agg.aggregate(as, workers), problem)
end

"""
    (::AggregationAlgorithm{Q,A})(q::Q, problem::Any) where {Q,A}

Steps performed by the workers when they receive a query `x::Q` from the central node
"""
function (agg::AggregationAlgorithm{Q,A})(q::Q, problem::Any) where {Q,A}
    agg.answer(q, problem)
end



"""
    AveragingAlgorithm{Q,A}(args...; kwargs...) -> AveragingAlgorithm{Q,A} <: AbstractAlgorithm{Q,A}

Distributed algorithm that writes: `q_j <- query(weighted_average([answer(q_i) for i in connected]))`
Where a "connected" worker is a worker that has answered at least once.
(Memory optimized: only the equivalent of one answer is stored on the central worker at all times)

# Arguments
- `initialize::Function`: with signature `initialize(problem::Any)`
- `query::Function`: with signature `query(a::A, problem::Any)`
- `answer::Function`: with signature `answer(q::Q, problem::Any)`

# Keywords
- `pids::Vector{Int64}=workers()`: `pids` of the active workers
- `weights::Vector{Float64}=ones(length(pids))`: weight of each pid in the weighted average
"""
mutable struct AveragingAlgorithm{Q,A} <: AbstractAlgorithm{Q,A}
    pids::Vector{Int64}
    initialize::Function
    query::Function
    answer::Function
    connected::BitVector
    last_normalization::Float64
    last_answer::Union{A,Nothing}
    last_answers::Vector{A}
    last_average::Union{A,Nothing}
    weights::Vector{Float64}
    function AveragingAlgorithm{Q,A}(initialize::Function, query::Function, answer::Function; pids=procs(), weights=ones(nprocs())) where {Q,A}
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

        new(pids, initialize, query, answer, connected, last_normalization, last_answer, last_answers, last_average, sparse_weights)
    end
end

"""
    (::AveragingAlgorithm{Q,A})(problem::Any) where {Q,A}

Initialization step: computing the first iterate 
"""
function (avg::AveragingAlgorithm{Q,A})(problem::Any) where {Q,A}
    avg.initialize(problem)
end

"""
    (::AveragingAlgorithm{Q,A})(δas::Vector{A}, workers::Vector{Int64}, problem::Any) where {Q,A}

Asynchronous step performed by the central node when receiving an answer `a::A` from a worker.
"""
function (avg::AveragingAlgorithm{Q,A})(δa::A, worker::Int64, problem::Any) where {Q,A}
    avg.connected[worker] = true
    normalization = sum(avg.connected .* avg.weights)
    avg.last_average = isnothing(avg.last_average) ? δa : (avg.weights[worker] * δa + avg.last_normalization * avg.last_average) / normalization
    avg.last_normalization = normalization
    avg.query(avg.last_average, problem)
end

"""
    (::AveragingAlgorithm{Q,A})(δas::Vector{A}, workers::Vector{Int64}, problem::Any) where {Q,A}

Synchronous step performed by the central node when receiving answers `a::Vector{A}` respectively from `workers::Vector{Int64}`
"""
function (avg::AveragingAlgorithm{Q,A})(δas::Vector{A}, workers::Vector{Int64}, problem::Any) where {Q,A}
    avg.last_average = isnothing(avg.last_average) ? δas : sum(avg.weights[workers] * δas) / sum(avg.weights) + avg.last_average
    avg.query(avg.last_average, problem)
end

"""
 (::AveragingAlgorithm{Q,A})(q::Q, problem::Any) where {Q,A}

Steps performed by the workers when they receive a query `q::Q` from the central node
"""
function (avg::AveragingAlgorithm{Q,A})(q::Q, problem::Any) where {Q,A}
    a = avg.answer(q, problem)
    δa = isnothing(avg.last_answer) ? a : a - avg.last_answer
    avg.last_answer = a
    return δa
end

