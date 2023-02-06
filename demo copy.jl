## Usage

# Launch multiple processes (or remote machines)
using Distributed; addprocs(5)

# Instantiate and precompile environment in all processes

@everywhere (using Pkg; Pkg.activate(@__DIR__); Pkg.instantiate(); Pkg.precompile())

# You can now use AsynchronousIterativeAlgorithms
@everywhere include("src/AsynchronousIterativeAlgorithms.jl")
@everywhere (using .AsynchronousIterativeAlgorithms; const AIA = AsynchronousIterativeAlgorithms)

# Say you want to implemtent a distributed version Stochastic Gradient Descent
@everywhere begin
    # Create a algorithm subtyping AbstractAlgorithm{Q,A}
    mutable struct SGD<:AbstractAlgorithm{Vector{Float64},Vector{Float64}}
        stepsize::Float64
        previous_q::Vector{Float64} # previous query
        SGD(stepsize::Float64) = new(stepsize, Vector{Float64}())
    end

    # Initialisation: computing the first iterate 
    function (algorithm::SGD)(problem::Any)
        algorithm.previous_q = rand(problem.m)
    end

    # Define the step perfromed by the workers when they receive a query `q::Q` from the central node
    function (algorithm::SGD)(q::Vector{Float64}, problem::Any) 
        algorithm.stepsize * problem.∇f(q, rand(1:problem.n))
    end
    
    # And the asynchronous step performed by the central node when receiving an answer `a::A` from `worker`
    function (algorithm::SGD)(a::Vector{Float64}, worker::Int64, problem::Any) 
        algorithm.previous_q -= a
    end
end

# Let's test our algorithm on a linear regression problem 
@everywhere begin
    # Create an problem compatible with your algorithm 
    # Here that means providing the number of data points `n`, the dimemtion `m` of the regressor, and the ith componant of the gradient of the objective `∇f(x::Vector{Float64}, i::Int64)` 
    struct LRMSE
        A::Union{Matrix{Float64}, Nothing}
        b::Union{Vector{Float64}, Nothing}
        n::Int64
        m::Int64
        L::Float64
        # f::Function
        ∇f::Function
    end

    function LRMSE(A::Matrix{Float64}, b::Vector{Float64})
        n, m = size(A)
        L = maximum(A'*A)
        # f(x) = sum((A * x - b).^2) / n / 2
        ∇f(x) = A' * (A * x - b) / n
        ∇f(x,i) = A[i,:] * (A[i,:]' * x - b[i]) # data point i
        LRMSE(A, b, n, m, L, ∇f)
    end
end

# Provide the stopping criteria 
stopat = (1000,0,0.) # (iterations, epochs, time)

# Instanciate your algorithm 
algorithm = SGD(0.01)

# Create a function that returns an instance of your problem for a given pid
problem_constructor = (pid) -> LRMSE(rand(42,10),rand(42))

# And you can run!
history = run(algorithm, problem_constructor, stopat);

## Working with a distributed problem

# When instanciating your problems you might have three requirement:

# - **Limiting comunication costs** and **avoiding duplicated memory**: therefore loading the problems directly on their correct processes is be preferable to loading them on the central node before sending each of them to their process
# - **Persistant data**: You might want to reuse problems you've created for other experiments

# Depending on your needs, you have three options to construct your problems: 

# |  	| communication costs <br>& duplicated memory 	| single use problems  	|
# |---	|:---:	|:---:	|
# | Option 1 	|  	| ❌ 	|
# | Option 2 	| ❌ 	|  	|
# | Option 3 	|  	|  	|

# Suppose you have a `make_problem` function

# Here we could also be reading the `A` and `b` from a file for example
@everywhere function make_problem(pid)
    pid==1 && return nothing # for now let's give process 1 an empty problem
    return LRMSE(rand(pid,10),rand(pid)) # here the sample size n == pid, don't do this irl
end

# Option 1: Instantiate the problems remotely at each process
problem_constructor = (pid) -> return make_problem(pid)

# Option 2: Instantiate the problems on the central node and send them to the problems
problems = Dict(procs() .=> make_problem.(procs()));
problem_constructor = (pid) -> return problems[pid]

# Option 3: Create a `DistributedObject` that references a problem on each active process. This last option uses [`DistributedObjects`](https://github.com/Selim78/DistributedObjects.jl) (In a nutshell, a `DistributedObject` instance references at most one object per process, and you can access the object stored on the current process with `[]`)
@everywhere using DistributedObjects
distributed_problem = DistributedObject((pid) -> make_problem(pid), pids=procs())
# problem_constructor = (pid) -> return distributed_problem[]



# As we saw *Option 2* should be discarded if you're working with large data, however it does allows you to have access to the **global problem**, which *Option 1* doesn't

function LRMSE(problems::Dict)
    pids = [pid for pid in keys(problems) if pid ≠ 1]
    n = sum([problems[pid].n for pid in pids])
    m = problems[pids[1]].m
    L = sum([problems[pid].L for pid in pids])
    # f(x) = sum([problems[pid].f(x) * problems[pid].n for pid in pids]) / n 
    ∇f(x) = sum([problems[pid].∇f(x) * problems[pid].n for pid in pids]) / n
    return LRMSE(nothing,nothing,n,m,L,∇f)
end

problems[1] = LRMSE(problems);
algorithm = SGD(1/problems[1].L)

# *Option 3* is the best of both worlds

function LRMSE(d::DistributedObject)
    pids = [pid for pid in where(d) if pid ≠ 1]
    n = sum(fetch.([@spawnat pid d[].n for pid in pids]))
    m = fetch(@spawnat pids[1] d[].m)
    L = sum(fetch.([@spawnat pid d[].L for pid in pids]))
    # f(x) = sum(fetch.([@spawnat pid d[].f(x) * d[].n for pid in pids])) / n 
    ∇f(x) = sum(fetch.([@spawnat pid d[].∇f(x) * d[].n for pid in pids])) / n
    return LRMSE(nothing,nothing,n,m,L,∇f)
end

distributed_problem[] = LRMSE(distributed_problem);
algorithm = SGD(1/distributed_problem[].L)

## Synchronous run
# If you want to run your algorithm synchronously you just have to define a new method with following signature
@everywhere begin
    # Synchronous step performed by the central node when receiving an answer `x::A` from a worker
    (algorithm::SGD)(xs::Vector{Vector{Float64}}, workers::Vector{Int64}, problem::Any) = sum(xs)
end
# and to add the `synchronous=true` keyword to run
history = run(algorithm, distributed_problem, stopat; synchronous=true);

## Active processes
# You can chose which processes are active with the `pid` keyword
history = run(algorithm, problem_constructor, stopat; pids=[2,3,6]);

# You can run a non-distributed (and necessarily synchronous) version of your algorithm by passing `[1]` to `pids`
history = run(algorithm, (pid)->LRMSE(rand(42,10),rand(42)), stopat; pids=[1], synchronous=true);

## Recording query iterates 
# The query`::Q` iterates sent by the central nodes are saved every `iterations`, `epochs` in `saveat=(iterations, epochs)`
history = run(algorithm, problem_constructor, stopat; saveat=(10,0));

# You can save the answers`::A` made by the workers by adding the `save_answers=true` keyword 
history = run(algorithm, problem_constructor, stopat; saveat=(10,0), save_answers=true);

## Custom stopping criterion 
# To **add** a stopping criterion to the default `stopat=(iteration, epoch, time)` you just need to define a new method `AsynchronousIterativeAlgorithms.stopnow` for dispatch on your algorithm and declare that your algorithm implements the `Stoppable` trait. 
# Let's modify the `SGD` example a bit to add a precision criterion

@everywhere begin
    using LinearAlgebra

    mutable struct CustomSGD<:AbstractAlgorithm{Vector{Float64},Vector{Float64}}
        stepsize::Float64
        previous_q::Vector{Float64}
        gap::Float64
        precision::Float64
        CustomSGD(stepsize::Float64, precision) = new(stepsize, Vector{Float64}(), 10^6, precision)
    end

    function (algorithm::CustomSGD)(problem::Any) 
        algorithm.previous_q = rand(problem.m)
    end

    function (algorithm::CustomSGD)(q::Vector{Float64}, problem::Any)
        return algorithm.stepsize * problem.∇f(q, rand(1:problem.n))
    end

    function (algorithm::CustomSGD)(a::Vector{Float64}, worker::Int64, problem::Any) 
        q = algorithm.previous_q - a 
        algorithm.gap = norm(q-algorithm.previous_q)
        algorithm.previous_q = q
    end

    # Stop when gap is small enough
    AIA.stopnow(algorithm::CustomSGD) = algorithm.gap ≤ algorithm.precision
    AIA.Stoppability(::CustomSGD) = Stoppable()    
end

history = run(CustomSGD(0.01, 0.1), distributed_problem, (10,0,0.); saveat=(0,0), verbose=1, save_answers=true, synchronous=false);

# Note that you can ask for a precision threshold by passing a fourth value in `stopat`. If you do so, and if you don't want to use `(x,y)->norm(x-y)`, make sure to pass your distance function to the `distance` keyword of run.
history = run(CustomSGD(0.01, 0.1), distributed_problem, (10,0,0., 0.1); saveat=(0,0), verbose=1, save_answers=true, synchronous=false);


history = run(CustomSGD(0.01, 0.1), distributed_problem, (0,0,0.); saveat=(0,0), verbose=1, save_answers=true, synchronous=false);
history

## `run!`
# By default, the algorithm you pass to `run` isn't modified. If you want it to be modified (to record some information during the optimization for instance) you can use `run!`

problem = problem_constructor(myid())
query = algorithm(answer, worker, problem) 
## Algorithm template 

# You are free to create your own algorithms. But you might be interested in *aggregation algorithms*, you might not have to implement everything from scratch.
# $$q_j <- \textrm{query}(\textrm{aggregate}_{i in \textrm{connected}}(a_j)) \textrm{where} a_i = \textrm{answer}(q_i)$$ 
# where `q_j` is computed by the worker uppon reception of `answer(q_j)` from worker `j` and where `connected` are the list of worker that have answered.

# Memory limitation: At any point in time, the central worker should have access to the most recent answers `a_i` from all the connected workers. This means storing a lot of `a_i`s if we use many workers.
# There is a workaround when the aggregation operation is an *average*. In this case only the equivalent of one answer needs to be saved on the central node, no matter the number of workers.
# This library proposes two templates, `AggregationAlgorithm` which isn't memory rund but works for any operation `aggregate` and `AverageAlgorithm` which is memory rund.
   
# To take advantage og `AggregationAlgorithm` you only need to specify the `query`, the `answer` and `aggregate`. Here's an example which shows the required signature 
@everywhere begin 
    using Statistics

    function agg_gd(q0, stepsize)
        initialize(problem::Any) = q0
        aggregate(a::Vector{Vector{Float64}}, connected::Vector{Int64}) = mean(a)            
        query(a::Vector{Float64}, problem::Any) = a
        answer(q::Vector{Float64}, problem::Any) = (q - stepsize * problem.∇f(q), throw(error("Oye")))

        AggregationAlgorithm{Vector{Float64}, Vector{Float64}}(initialize, aggregate, query, answer; pids=workers())
    end
end 


# To use `AverageAlgorithm` you only define `query`, the `answer` (with the same signature) 
@everywhere begin 
    # If you want the average to be weighted, you can add the keywords pids with their corresponding weights
    function avg_gd(q0, stepsize, pids=workers(), weights=ones(nworkers())) 
        initialize(problem::Any) = q0
        query(a::Vector{Float64}, problem::Any) = a
        answer(q::Vector{Float64}, problem::Any) =  q - stepsize * problem.∇f(q)
        AveragingAlgorithm{Vector{Float64}, Vector{Float64}}(initialize, query, answer; pids=pids, weights=weights)
    end
end

agg_gd_algorithm = agg_gd(rand(10), 0.01);
avg_gd_algorithm = avg_gd(rand(10), 0.01);

history = run(agg_gd_algorithm, distributed_problem, (1000,0,0.); saveat=(1,0), verbose=1, save_answers=true, synchronous=false);


using InteractiveUtils 
@code_warntype run(SGD(0.02), distributed_problem, stopat, verbose=1)


@code_warntype run(agg_gd_algorithm, distributed_problem, stopat, synchronous=true)
@code_warntype run(agg_gd_algorithm, distributed_problem, stopat, pids=[1], synchronous=true)

# using BenchmarkTools    
# @btime history = run(algorithm_1, distributed_problem, stopat)
# @btime history = run(algorithm_2, distributed_problem, stopat)
# using Profile
# using PProf
# @profile run(algorithm_1, distributed_problem, stopat)
# @profile run(algorithm_1, distributed_problem, stopat, synchronous=true)

# pprof()


 # # Just for the convenience of broadcasting algorithms
# Base.broadcastable(s::AbstractAlgorithm{Q,A}) where {Q,A} = Ref(s)




# args and kwargs for each experiments
parameters = Dict("AVG" => (
                                (avg_gd(rand(10), 0.01), 
                                distributed_problem, 
                                (0, 100, 0.)),

                                (saveat=(0,100), 
                                save_answers=true, 
                                pids=workers(), 
                                synchronous=false, 
                                verbose=0)
                                ),
                 "AGG" => (
                                (agg_gd(rand(10), 0.01),  
                                distributed_problem, 
                                (0, 100, 0.)),

                                (saveat=(0,100), 
                                save_answers=false, 
                                pids=workers(), 
                                synchronous=false, 
                                verbose=0)
                            ),
                 "SGD" => (
                                (SGD(0.01),  
                                distributed_problem, 
                                (0, 100, 0.)),

                                (saveat=(0,100), 
                                save_answers=false, 
                                pids=workers(), 
                                synchronous=false, 
                                verbose=0)
                            )

                    )

begin
    using ProgressMeter

    function experiment(parameters, repeat=5)
        histories = Dict()
        @showprogress 0.1 "Experimenting:" for (name, (args, kwargs)) in parameters
            histories[name] = [run(args...; kwargs...) for _ in 1:repeat]
        end
        histories
    end
    histories = experiment(parameters)

    function compute(histories, f, symbol)
        for name in keys(histories)
            histories[name] = [merge(history, NamedTuple{(symbol,)}([f(history)])) for history in histories[name]]
        end
        histories
    end

    norm∇f(query) = distributed_problem[].∇f(query) .^2 |> sum
    computed_history = compute(histories, (history)->norm∇f.(history.queries), :norm∇f)


    using Statistics
    using Plots

    struct PiecewiseLinear
        X::Vector
        Y::Vector
        function PiecewiseLinear(X::Vector,Y::Vector)
            p = sortperm(X)
            new(X[p],Y[p])
        end
    end

    function (pl::PiecewiseLinear)(x)
        i = findfirst(x .== pl.X); !isnothing(i) && return pl.Y[i]
        i = searchsortedfirst(pl.X, x)
        return pl.Y[i-1] + (pl.Y[i]-pl.Y[i-1])*(x - pl.X[i-1])/(pl.X[i] - pl.X[i-1]) 
    end

    function average_points(X, Y)
        @assert (n = length(X)) == length(Y)

        window = [maximum(minimum.(X)), minimum(maximum.(X))]
        crop(x) = x[window[1] .≤ x .≤ window[2]]
        x = sort(crop(vcat(X...)))

        PiecewiseLinear(X[1], Y[1])

        Y = [PiecewiseLinear(X[i], Y[i]).(x) for i in 1:n]
        return x, mean(Y), std(Y)
    end


    function average_plots(histories, xaxis, yaxis)
        graphs = Dict()
        for name in keys(histories)
            X = [history[xaxis] for history in histories[name]]
            Y = [history[yaxis] for history in histories[name]]
            x, y, v = average_points(X, Y)
            graphs[name] = [x, y, v]
        end
        graphs
    end

    graphs = average_plots(computed_history, :epochs, :norm∇f);

    plots = Dict()
    for (name, (x, y, v)) in graphs
        plots[name] = plot(x, y, ribbon=v, title=name)
    end

    plot(values(plots)...)
end
@show 1

# function average_plots(histories, axes...)
#     Xs = [[(x = xy[1]; history.x) for history in histories] for xy in axes]
#     Ys = [[(y = xy[2]; history.y) for history in histories] for xy in axes]

#     output = []
#     for (X, Y) in zip(Xs, Ys) # for each axis
#         x, y, v = average_points(X, Y) # average over histories
#         output.append([[x, y, v]])
#     end
#     output
# end

# function experiment(args...)
    
#     histories = Vector{NamedTuple}(undef, length(args))
#     arg_make_data = nothing
    
#     for (i, arg) in enumerate(args)
#         if arg_make_data != make_data(arg["make_data"])
#             arg_make_data = make_data(arg["make_data"]) 
#             @everywhere distributed_problem = DistributedObject((pid)->LRMSE(make_data(arg_make_data...,pid)...)) 
#             problem_constructor = (pid) -> distributed_problem[]
#         end
#         @everywhere algorithm = GradientDescent{Vector{Float64},Vector{Float64}}(arg["algorithm"]...)
#         histories[i] = run(algorithm, problem_constructor, arg["run"]...)
#     end
#     histories
# end 

# function plot_experiments(histories) end



x1 = randn(10)*10
x2 = randn(15)*10

y1 = randn(10)*10
y2 = randn(15)*10

X = [x1, x2]
Y = [y1, y2]



x, y, v = average_points(X, Y)

plot(x, y, ribbon=v, fillalpha=.3) #, label=label, linestyle=linestyle, color=color)  
®
ok
# """
    #     PartialAggregationAlgorithm{Q,A}(partial_aggregate::Function, query::Function, answer::Function, initial_answer::A; pids=workers()) where Q where A

    # Create a algorithm for distributed algorithms that write:
    
    # At central node:
    # ```julia
    # initialise a with initial_aggregate
    # loop
    #     δa_i <— receive δa_i from worker j
    #     a <- partial_aggregate(δa_i, a)
    #     q_j <— query(a)
    #     send q_j to worker j
    # end
    # ```

    # At worker node `j`:
    # ```julia
    # loop
    #     receive q_j from 0
    #     send partial_answer(answer(q_j), a_i) to 0 
    #     a_i <- answer(q_j)
    # end
    # ```

    # The parameters should have the following signature 
    # - `partial_aggregate(δa::A, a::A, worker::Int64, last_connected::BitVector, connected::BitVector)`
    # - `query(a::A, problem::Any)`
    # - `answer(q::Q, problem::Any)`
    # - `answer(q::Q, problem::Any)`
    # - `initial_answer::A` the initiale value of `a` on every worker
    # """
    # struct PartialAggregationAlgorithm{Q,A}<:AbstractAlgorithm{Q,A}
    #     pids::Vector{Int64}
    #     connected::Vector{Int64}
    #     partial_aggregate::Function
    #     query::Function
    #     answer::Function
    #     aggregated_answers::Union{A, Nothing}
    #     last_a::A
    #     function PartialAggregationAlgorithm{Q,A}(partial_aggregate::Function, query::Function, answer::Function, initial_answer::A; pids=workers()) where Q where A
    #         connected = BitVector([false for pid in pids])
    #         last_a = initial_answer
    #         aggregated_answers = nothing
    #         new(pids, connected, partial_aggregate, query, answer, aggregated_answers, last_a)
    #     end
    # end

    # # Step performed by the central node when receiving an answer `x::A` from a worker
    # function (s::PartialAggregationAlgorithm{Q,A})(δa::A, worker::Int64, problem::Any) where Q where A
    #     s.connected[worker] = true
    #     s.aggregated_answers = s.partial_aggregate(δa, s.aggregated_answers, worker, s.connected) 
    #     s.queries[worker] = s.query(s.aggregated_answers, problem)
    # end

    # # Steps perfromed by the wokers when they receive a query `x::Q` from the central node
    # function (algorithm::PartialAggregationAlgorithm{Q,A})(q::Q, problem::Any) where Q where A
    #     a = answer(q, problem)
    #     δa = a - s.last_a
    #     s.last_a = a
    #     return δa
    # end



    # At central node:
    # ```julia
    # initialize a_i for i in pids with initial_answer
    # send a_i to workers i for i in pids
    # loop
    #     a_j <— receive a_j from worker j
    #     q_j <— query(aggregate([a_i for i in pids]))
    #     send q_j to worker j
    # end
    # ```
    # At worker node `j`:
    # ```julia
    # loop
    #     receive q_j from 0
    #     send answer(q_j) to 0 
    # end
    # ```



weights = [3.,10,1]
pids = [2,4,5]
maxpid = 5

weights = zeros(maxpid)
for (pid, weight) in zip(pids, weights)
    weights[pid] = weight
end





# forward methods on a struct to a field of that struct.
# good for your composition.
# syntax: @forward CompositeType.property Base.iterate Base.length :*
# Symbol literals automatically become Base.:symbol. Good for adding
# methods to built-in operators.
macro forward(property, functions...)
    structname = property.args[1]
    field = property.args[2].value
    block = quote end
    for f in functions
        # case for operators
        if f isa QuoteNode
            f = :(Base.$(f.value)) 
            def1 = :($f(x::$structname, y) = $f(x.$field, y))
            def2 = :($f(x, y::$structname) = $f(x, y.$field))
            push!(block.args, def1, def2)
        # case for other stuff
        else
            def = :($f(x::$structname, args...; kwargs...) = $f(x.$field, args...; kwargs...))
            push!(block.args, def)
        end
    end
    esc(block)
end

# demo:
struct Foo
    x::String
end

methodswith(String)[2].sig

@forward Foo.x Base.string :* :^

struct HasInterestingField
    data::String
end

double(hif::HasInterestingField) = hif.data ^ 2
shout(hif::HasInterestingField) = uppercase(string(hif.data, "!"))
 
# the compositional way add those fields to another struct.
struct WantsInterestingField
    interesting::HasInterestingField
end


function WantsInterestingField(data) 
    wif = WantsInterestingField(HasInterestingField(data))
    # forward methods
    for method in (:double, :shout)
        @eval $method(wif::WantsInterestingField) = $method(wif.interesting)
    end
    return wif
end
 


# same as:
#     double(wif::WantsInterestingField) = double(wif.interesting)
#     shout(wif::WantsInterestingField) = shout(wif.interesting)
wif = WantsInterestingField("foo")
@show shout(wif)
@show double(wif);
     


x= [1,2,3]

struct Histo
    h::Vector{Vector{Int64}}
end 

hist = Histo([])
hist.h
append!(hist.h, [x])
append!(hist.h, [x])

x[2] = 234

copy(true)

ismutable(hist)

Base.copymutable(hist)



# Here's a very stupid algorithm:
# Each worker averages the color it reveives with it's `problem`
# The central node averages its color with the one it receives  
@everywhere begin
    # Create a algorithm subtyping AbstractAlgorithm{Q,A}
    mutable struct Writting{Q,A}<:AbstractAlgorithm{Q,A}
        book::String
    end

    # Step perfromed by the workers when they receive a query `x::Q` from the central node
    function (algorithm::Writting{Q,A})(x::Q, problem::Any) where Q where A
        x * string(problem)
    end

    # Depending on you needs, define one or both of the following: 

    # Asynchronous step performed by the central node when receiving an answer `x::A` from a worker
    function (algorithm::Writting{Q,A})(x::A, worker::Int64, problem::Any) where Q where A
        algorithm.book *= string(" -- $(worker) says: ") * x
    end

    # Synchronous step performed by the central node when receiving an answer `x::A` from a worker
    function (algorithm::Writting{Q,A})(xs::Vector{A}, workers::Vector{Int64}, problem::Any) where Q where A
        algorithm.book *= string(" -- $(workers...) say:") * string(xs...)
    end
end

# Create an problem compatible with your algorithm (here we should be able to call `string` on the problem) 
@everywhere begin
    struct Plant
        name::Union{String,Nothing}
        edible::Union{Bool, Nothing}
    end

    Base.string(p::Plant) = "$(p.name) $(p.edible ? "is" : "isn't") edible"
end

# Here's a very stupid algorithm:
# Each worker averages the color it reveives with it's `problem`
# The central node averages its color with the one it receives  
@everywhere begin
    # Create a algorithm subtyping AbstractAlgorithm{Q,A}
    mutable struct Writting{Q,A}<:AbstractAlgorithm{Q,A}
        book::String
    end

    # Step perfromed by the workers when they receive a query `x::Q` from the central node
    function (algorithm::Writting{Q,A})(x::Q, problem::Any) where Q where A
        x * string(problem)
    end

    # Depending on you needs, define one or both of the following: 

    # Asynchronous step performed by the central node when receiving an answer `x::A` from a worker
    function (algorithm::Writting{Q,A})(x::A, worker::Int64, problem::Any) where Q where A
        algorithm.book *= string(" -- $(worker) says: ") * x
    end

    # Synchronous step performed by the central node when receiving an answer `x::A` from a worker
    function (algorithm::Writting{Q,A})(xs::Vector{A}, workers::Vector{Int64}, problem::Any) where Q where A
        algorithm.book *= string(" -- $(workers...) say:") * string(xs...)
    end
end

# Create an problem compatible with your algorithm (here we should be able to call `string` on the problem) 
@everywhere begin
    struct Plant
        name::Union{String,Nothing}
        edible::Union{Bool, Nothing}
    end

    Base.string(p::Plant) = "$(p.name) $(p.edible ? "is" : "isn't") edible"
end

using ProgressMeter
progress_meter = Progress(11; desc="Optimizing:", showspeed=true)
function generate_showvalues(i) 
    # answer_count = join(["$key\u0009$(h.answer_count[key])" for key in sort!(collect(keys(h.answer_count)))], "\n\u0009\u0009")
    () -> [(:iteration, i), (:answers, string("asadksjndakjdnkasjd")[6:end-1])]  
end

for i in 1:10
    sleep(0.2)
    update!(progress_meter, i, showvalues = generate_showvalues(i))
end
update!(progress_meter, 11, showvalues = generate_showvalues(11))

@everywhere function my_test()
    task = @task (println(2); throw(error("heeere")))
    schedule(task)
end

my_test()

remote_do(my_test, 2)

remotecall_fetch(my_test, 2)


struct AA 
    a
end

AA() = nothing

using InteractiveUtils 

d = Dict{Int64, Int64}(1=>5, 3=>54, 324=>54)
@code_warntype d[1, 3, 324]

function do_errors()
    try 
        error("Ooops")
    catch e
        @show e
        rethrow(e)
    finally 
        @show "ok"
        2
    end
end

a = do_errors()

a




struct RecordedAlgorithm{A,B}
    algorithm<:AbstractAlgorithm{A,B}
    history::History{A,B}
end

function (r::RecordedAlgorithm{Q,A})(problem::Any) where {Q,A} 
    r.algorithm(problem)
end

function (r::RecordedAlgorithm{Q,A})(q::Q, problem::Any) where {Q,A} 
    r.algorithm(q,problem)
end
function (r::RecordedAlgorithm{Q,A})(a::A, worker::Int64, problem::Any) where {Q,A} 
    r.algorithm(a,worker,problem)
end




-`history(algorithm::MyAlgorithm)` (with the trait `historyability(::MyAlgorithm) = historyable()`) which will replace the default output of `run`

- `Union{NamedTuple, Dict}`: a history of the optimization 
    - by default `nt::NamedTuple` which records the `queries` and the `iterations`, `epochs`, `timestamps` at which they were recorded, as well as `answer_count` of each worker (if `save_answers` is `true`, the `answers` will be recorded with their worker provenance in `answer_origin`)
    - if `historyability(::MyAlgorithm) = historyable()`, a dictionary `Dict("default"->nt, "custom"->history(algorithm))`
    -`history(algorithm::MyAlgorithm)` (with the trait `historyability(::MyAlgorithm) = historyable()`) which will replace the default output of `run`
# ## Custom history 
# # If you want more information from the optimization than the default `(queries=..., answers=..., iterations=..., epochs=..., timestamps=..., answer_origin=..., answer_count=...)` you can define a new method `AIA.history` for dispatch on your algorithm and declare that your algorithm implement the `historyable` trait.
# # The output will then be a `Dict("default"->(...), "custom"->(...))` 

# @everywhere begin
#     AIA.history(algorithm::CustomSGD) = (gaps=algorithm.gaps)
#     AIA.historyability(::CustomSGD) = historyable()
# end

# history(rs::RecordedAlgorithm{Q,A}) where {Q,A} = history(historyability(rs.algorithm), rs)
# function history(::historyable, rs) 
#     Dict("default" => (queries=rs.queries, answers=rs.answers, iterations=rs.iterations, epochs=rs.epochs, timestamps=rs.timestamps, answer_origin=rs.answer_origin, answer_count=rs.answer_count),
#          "custom" => history(rs.algorithm))
# end
# function history(::Nothistoryable, rs::RecordedAlgorithm{Q,A}) where {Q,A}
#     (queries=rs.queries, answers=rs.answers, iterations=rs.iterations, epochs=rs.epochs, timestamps=rs.timestamps, answer_origin=rs.answer_origin, answer_count=rs.answer_count)
# end