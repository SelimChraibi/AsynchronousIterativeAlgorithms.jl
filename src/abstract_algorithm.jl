using Distributed
using ProgressMeter



export AbstractAlgorithm, stopnow, savenow, savevalues, report, progress, showvalues

"""
    AbstractAlgorithm{Q,A}

To be compatible with [`start`](@ref), types subtyping `AbstractAlgorithm` should be callable with the following signatures:
- `(algorithm::AbstractAlgorithm{Q,A})(problem::Any)::Q where {Q,A}`: the initialization step that create the first query iterate
- `(algorithm::AbstractAlgorithm{Q,A})(q::Q, problem::Any)::A where {Q,A}`: the answer step perfromed by the wokers when they receive a query `q::Q` from the central node
- `(algorithm::AbstractAlgorithm{Q,A})(a::A, worker::Int64, problem::Any)::Q where {Q,A}`: the query step performed by the central node when receiving an answer `a::A` from a worker
- when [`start`](@ref) takes the keyword `synchronous=true`, `(algorithm::AbstractAlgorithm{Q,A})(as::Vector{A}, workers::Vector{Int64}, problem::Any)::Q where {Q,A}`: the query step performed by the central node when receiving the answers `as::Vector{A}` respectively from the `workers`
"""
abstract type AbstractAlgorithm{Q,A} end

(algorithm::AbstractAlgorithm{Q,A})(problem::Any) where {Q,A} = throw(ArgumentError("Method (::$(typeof(algorithm)))(problem::Any) not implemented."))
(algorithm::AbstractAlgorithm{Q,A})(q::Q, problem::Any) where {Q,A} = throw(ArgumentError("Method (::$(typeof(algorithm)))(q::$Q, problem::Any) not implemented."))
(algorithm::AbstractAlgorithm{Q,A})(a::A, worker::Int64, problem::Any) where {Q,A} = throw(ArgumentError("Method (::$(typeof(algorithm)))(a::$A, worker::Int64, problem::Any) not implemented."))
(algorithm::AbstractAlgorithm{Q,A})(as::Vector{A}, workers::Vector{Int64}, problem::Any) where {Q,A} = throw(ArgumentError("Method (::$(typeof(algorithm)))(as::Vector{$A}, workers::Vector{Int64}, problem::Any) not implemented."))


mutable struct RecordedAlgorithm{Q,A,Alg<:AbstractAlgorithm{Q,A}} <: AbstractAlgorithm{Q,A}

    algorithm::Alg
    pids::Vector{Int64}

    iteration::Int64
    epoch::Int64 
    time::Float64
    iterations::Vector{Int64}
    epochs::Vector{Int64}
    timestamps::Vector{Float64}

    queries::Vector{Q}
    answers::Vector{A}
    save_answers::Bool
    answer_count::Dict{Int64, Int64}
    answers_origin::Vector{Int64}
    
    start_time::Float64
    progress_meter::Progress
    stopat::NamedTuple
    saveat::NamedTuple
    verbose::Int64
    
    function RecordedAlgorithm(algorithm::Alg, stopat::NamedTuple, saveat::NamedTuple, pids::Vector{Int64}, save_answers::Bool, verbose::Int64) where Alg<:AbstractAlgorithm{Q,A} where {Q,A}
        algorithm = deepcopy(algorithm)
        pids = sort(pids)
        
        iteration = 0
        epoch = 0
        time = 0.
        iterations = Int64[]
        epochs = Int64[]
        timestamps = Float64[]

        queries = Q[]
        answers = A[]
        answer_count = Dict(pids .=> 0)
        answers_origin = Int64[]
        
        start_time = 0.
        progress_meter = Progress(100; desc="Iterating:", showspeed=true, enabled=verbose>0)

        new{Q,A,Alg}(algorithm, pids, iteration, epoch, time, iterations, epochs, timestamps, queries, answers, save_answers, answer_count, answers_origin, start_time, progress_meter, stopat, saveat, verbose)
    end
end

# Initialize
function (ra::RecordedAlgorithm)(problem::Any)
    q = ra.algorithm(problem)
    update(ra)
    savevalues(ra, q)
    return q
end

# Asynchronous central iteration 
function (ra::RecordedAlgorithm{Q,A,Alg})(a::A, worker::Int64, problem::Any) where {Q,A,Alg}
    q = ra.algorithm(a, worker, problem)
    update(ra, worker)
    savenow(ra) && savevalues(ra, q, a, worker)
    q
end

# Synchronous central iteration 
function (ra::RecordedAlgorithm{Q,A,Alg})(as::Vector{A}, workers::Vector{Int}, problem::Any) where {Q,A,Alg}
    q = ra.algorithm(as, workers, problem)
    update(ra, workers)
    savenow(ra) && savevalues(ra, q, as, workers)
    q
end

# Worker iteration 
function (ra::RecordedAlgorithm{Q,A,Alg})(q::Q, problem::Any) where {Q,A,Alg}
    ra.algorithm(q, problem)
end


# Update the current iteration, epoch, time, answer_count and display the progress [Initialisation]
function update(ra::RecordedAlgorithm)
    ra.iteration = 1
    ra.epoch = 1 
    ra.start_time = time_ns()
    ra.time = 0.
    update!(ra.progress_meter, 0, showvalues = ()->showvalues(ra))
end
# Update the current iteration, epoch, time, answer_count and display the progress [Asynchronous step]
function update(ra::RecordedAlgorithm, worker::Int64)
    ra.answer_count[worker] += 1
    ra.iteration += 1
    ra.time = (time_ns() - ra.start_time)/1e9
    all(values(ra.answer_count) .>= ra.epoch) && (ra.epoch += 1)
    update!(ra.progress_meter, floor(Int64, progress(ra)*100), showvalues = ()->showvalues(ra))
end
# Update the current iteration, epoch, time, answer_count and display the progress [Synchronous step]
function update(ra::RecordedAlgorithm, workers::Vector{Int64})
    foreach((worker)->ra.answer_count[worker] += 1, workers)
    ra.iteration += 1
    ra.epoch += 1
    ra.time = (time_ns() - ra.start_time)/1e9
    update!(ra.progress_meter, floor(Int64, progress(ra)*100), showvalues = ()->showvalues(ra))
end 

"""
    savevalues(::AbstractAlgorithm) = nothing

Define this method on your algorithm`<:AbstractAlgorithm`. It will be called at each iteration where [`savenow`](@ref) returns `true`: 
store some values on your algorithm object (don't forget to define [`report`](@ref) to retrieve what you stored)
"""
savevalues(::AbstractAlgorithm) = nothing

# Return true if the saving condition has been reached [Initialisation]
function savevalues(ra::RecordedAlgorithm{Q,A,Alg}, q::Q) where {Q,A,Alg}
    append!(ra.queries, [copy(q)])
    append!(ra.iterations, [ra.iteration])
    append!(ra.epochs, [ra.epoch])
    append!(ra.timestamps, [ra.time])
    savevalues(ra.algorithm)
end
# Return true if the saving condition has been reached [Asynchronous step]
function savevalues(ra::RecordedAlgorithm{Q,A,Alg}, q::Q, a::A, worker::Int64) where {Q,A,Alg}
    append!(ra.queries, [copy(q)])
    append!(ra.iterations, [ra.iteration])
    append!(ra.epochs, [ra.epoch])
    append!(ra.timestamps, [ra.time])
    if ra.save_answers
        append!(ra.answers, [copy(a)])
        append!(ra.answers_origin, [worker])
    end
    savevalues(ra.algorithm)
end
# Return true if the saving condition has been reached [Synchronous step]
function savevalues(ra::RecordedAlgorithm{Q,A,Alg}, q::Q, as::Vector{A}, workers::Vector{Int64}) where {Q,A,Alg}
    append!(ra.queries, [copy(q)])
    append!(ra.iterations, [ra.iteration])
    append!(ra.epochs, [ra.epoch])
    append!(ra.timestamps, [ra.time])
    if ra.save_answers
        append!(ra.answers, copy(as))
        append!(ra.answers_origin, workers)
    end
    savevalues(ra.algorithm)
end

"""
    savenow(::AbstractAlgorithm, saveat::NamedTuple) = false

Define this method on your algorithm`<:AbstractAlgorithm` to add saving stops: 
return true if your saving condition has been reached
"""
function savenow(::AbstractAlgorithm, saveat::NamedTuple) 
    false 
end

# Return true if the saving condition has been reached
function savenow(ra::RecordedAlgorithm)
    stopnow(ra) || 
    savenow(ra.algorithm, ra.saveat) ||
    (haskey(ra.saveat, :iteration) && ra.iteration % ra.saveat.iteration == 0 ) || 
    (haskey(ra.saveat, :epoch) && ra.epoch % ra.saveat.epoch == 0)
end


"""
    progress(::AbstractAlgorithm, stopat::NamedTuple) = 0.

Define this method on your algorithm`<:AbstractAlgorithm` to change the display of the progress bar: 
return how close the current step is to reaching your stopping requirement on a scale of 0 to 1
"""
progress(::AbstractAlgorithm, stopat::NamedTuple) = 0.

# How close the current step is to reaching the stopping requirement on a scale of 0 to 1 
function progress(ra::RecordedAlgorithm)
    stopnow(ra) && return 1 
    output = 0.
    total = 0
    haskey(ra.stopat, :iteration) && (output += ra.iteration / ra.stopat.iteration; total +=1)
    haskey(ra.stopat, :epoch) && (output += ra.epoch / ra.stopat.epoch; total +=1)
    haskey(ra.stopat, :time) && (output += ra.time / ra.stopat.time; total +=1)    
    total > 0 && (output /= total)
    output = 1 - (1 - progress(ra.algorithm, ra.stopat)) * (1 - output)
end

"""
    stopnow(::AbstractAlgorithm, stopat::NamedTuple) = false

Define this method on your algorithm`<:AbstractAlgorithm` to add a stopping criterion: 
return true if your stopping condition has been reached.
"""
stopnow(::AbstractAlgorithm, stopat::NamedTuple) = false

# Return true if the stopping condition has been reached
function stopnow(ra::RecordedAlgorithm)
    stopnow(ra.algorithm, ra.stopat) ||
    (haskey(ra.stopat, :iteration) && ra.iteration ≥ ra.stopat.iteration) || 
    (haskey(ra.stopat, :epoch) && ra.epoch ≥ ra.stopat.epoch) ||
    (haskey(ra.stopat, :time) && ra.time ≥ ra.stopat.time) #||    
end

"""
    showvalues(::AbstractAlgorithm) = Tuple{Symbol, Any}[]
    
Define this method on your algorithm`<:AbstractAlgorithm` to add a values to be displayed below the progress bar when `verbose>1`: 
return a `Tuple{Symbol, Any}` with those values.
"""
showvalues(::AbstractAlgorithm) = Tuple{Symbol, Any}[]

# Values to be displayed with the progress bar when `verbose>1`
function showvalues(ra::RecordedAlgorithm, synchronous=false)
    values = Tuple{Symbol, Any}[(:iterations, ra.iteration)]
    synchronous || (values = append!(values, [(:epochs, ra.epoch)]))
    values = append!(values, [(:answers, string((ra.pids .=> getindex.([ra.answer_count], ra.pids)))[2:end-1])])
    values = append!(values, showvalues(ra.algorithm))
    values
end

"""
    report(::AbstractAlgorithm) = NamedTuple()

Define this method on your algorithm`<:AbstractAlgorithm` to add custom values to the results outputted by [`start`](@ref): 
return a `NamedTuple()` with those values, making sure to not reuse the field names `queries`, `answers`, `iterations`, `epochs`, `timestamps`, `answers_origin`, `answer_count`.
"""
report(::AbstractAlgorithm) = NamedTuple()

# Compile the results to be outputted 
function report(ra::RecordedAlgorithm)
    merge((queries=ra.queries, answers=ra.answers, iterations=ra.iterations, epochs=ra.epochs, timestamps=ra.timestamps, answers_origin=ra.answers_origin, answer_count=ra.answer_count), report(ra.algorithm))
end
