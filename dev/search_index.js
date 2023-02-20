var documenterSearchIndex = {"docs":
[{"location":"manual/#Manual","page":"Manual","title":"Manual","text":"","category":"section"},{"location":"manual/","page":"Manual","title":"Manual","text":"We saw how to run an asynchronous version of the SGD algorithm on a LRMSE problem in quick start. Here we'll use this same example to look at the following:  ","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"Working with a distributed problem\nSynchronous run\nActive processes\nRecording iterates\nCustomization of start's execution\nHandling worker failures\nAlgorithm wrappers","category":"page"},{"location":"manual/#Working-with-a-distributed-problem","page":"Manual","title":"Working with a distributed problem","text":"","category":"section"},{"location":"manual/","page":"Manual","title":"Manual","text":"Suppose you have a make_problem function","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"# Note: In this example, we sample `A` and `b`. \n# In practice, we could read them from a file or any other source.\n@everywhere function make_problem(pid)\n    pid==1 && return nothing # for now, let's assign process 1 an empty problem\n    LRMSE(rand(pid,10),rand(pid)) # the sample size is `m` is set to `pid` for demonstration purposes only\nend","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"When instantiating your problems you might have three requirements:","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"Limiting communication costs and avoiding duplicated memory: loading problems directly on their assigned processes is preferable to loading them central node before sending them to their respective processes\nPersistent data: necessary if you want to reuse problems for multiple experiments (you don't want your problems to be stuck on  remote processes in start's local scope)","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"Depending on your needs, you have three options to construct your problems:","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"# Option 1: Instantiate the problems remotely\nproblem_constructor = make_problem \n\n# Option 2: Instantiate the problems on the central node and send them to their respective processes\nproblems = Dict(procs() .=> make_problem.(procs()));\nproblem_constructor = (pid) -> problems[pid]\n\n# Option 3: Create a `DistributedObject` that references a problem on each process. \n@everywhere using DistributedObjects\ndistributed_problem = DistributedObject((pid) -> make_problem(pid), pids=procs())","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"Option 3 uses DistributedObjects. In a nutshell, a DistributedObject instance references at most one object per process, and you can access the object stored on the current process with []","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":" communication costs & duplicated memory single use objectives \nOption 1  ❌ (Image: )\nOption 2 ❌  (Image: )\nOption 3   (Image: )","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"As previously noted, Option 2 should be avoided when working with large data. However, it does offer the advantage of preserving access to problems, which is not possible with Option 1. This opens up the possibility of reconstructing the global problem.","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"# reconstructing global problem from problems stored locally\nfunction LRMSE(problems::Dict)\n    pids = [pid for pid in keys(problems) if pid ≠ 1]\n    n = problems[pids[1]].n\n    m = sum([problems[pid].m for pid in pids])\n    L = sum([problems[pid].L for pid in pids])\n    ∇f(x) = sum([problems[pid].∇f(x) * problems[pid].m for pid in pids]) / m\n    return LRMSE(nothing,nothing,n,m,L,∇f)\nend\n\nproblems[1] = LRMSE(problems);\n# We now have access to the global Lipschitz constant!\nsgd = SGD(1/problems[1].L)","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"Option 3 is the best of both worlds:","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"# reconstructing global problem from problems stored remotely \nfunction LRMSE(d::DistributedObject)\n    pids = [pid for pid in where(d) if pid ≠ 1]\n    n = fetch(@spawnat pids[1] d[].n)\n    m = sum(fetch.([@spawnat pid d[].m for pid in pids]))\n    L = sum(fetch.([@spawnat pid d[].L for pid in pids]))\n    ∇f(x) = sum(fetch.([@spawnat pid d[].∇f(x) * d[].m for pid in pids])) / m\n    return LRMSE(nothing,nothing,n,m,L,∇f)\nend\n\ndistributed_problem[] = LRMSE(distributed_problem);\n# We also have access to the global Lipschitz constant!\nsgd = SGD(1/distributed_problem[].L)","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"It's worth mentioning that instead of problem_constructor::Function, distributed_problem::DistributedObject can be passed to start. Both of the following are equivalent:","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"history = start(sgd, (pid)-> distributed_problem[], stopat)\nhistory = start(sgd, distributed_problem, stopat);","category":"page"},{"location":"manual/#Synchronous-run","page":"Manual","title":"Synchronous run","text":"","category":"section"},{"location":"manual/","page":"Manual","title":"Manual","text":"If you want to run your algorithm synchronously you just have to define the synchronous central step performed by the central node when receiving answers as::Vector{A} from all the workers...","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"@everywhere begin\n    # synchronous central step\n    (sgd::SGD)(as::Vector{Vector{Float64}}, workers::Vector{Int64}, problem::Any) = sum(as)\nend","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"...and to add the synchronous=true keyword to start","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"history = start(sgd, distributed_problem, stopat; synchronous=true);","category":"page"},{"location":"manual/#Active-processes","page":"Manual","title":"Active processes","text":"","category":"section"},{"location":"manual/","page":"Manual","title":"Manual","text":"You can choose which processes are active with the pids keyword","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"history = start(sgd, problem_constructor, stopat; pids=[2,3,6]);","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"If pids=[1], a non-distributed (and necessarily synchronous) version of your algorithm will be started.","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"history = start(sgd, (pid)->LRMSE(rand(42,10),rand(42)), stopat; pids=[1], synchronous=true);","category":"page"},{"location":"manual/#recording_iterated","page":"Manual","title":"Recording iterates","text":"","category":"section"},{"location":"manual/","page":"Manual","title":"Manual","text":"The queries::Q sent by the central node, along with the iterations, epochs, times at wich they were recorded, are saved at intervals specified by the keyword saveat: every iteration and every epoch (see savenow for custom saving criteria).","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"You can set any or all criteria: saveat=(iteration=100, epoch=10) or saveat=(epoch=100,) for example.","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"To also save the workers' answers::A, simply add the save_answers=true keyword (see savevalues and report to save additional variables during execution).","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"history = start(sgd, distributed_problem, stopat; saveat=(iteration=100, epoch=10), save_answers=true);","category":"page"},{"location":"manual/#custom_execution","page":"Manual","title":"Customization of start's execution","text":"","category":"section"},{"location":"manual/","page":"Manual","title":"Manual","text":"Let's look at a slightly modified version of SGD where we track the \"precision\" of our iterative algorithm, measured as the distance between the last two iterates.","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"@everywhere begin\n    using LinearAlgebra\n\n    mutable struct SGDbis<:AbstractAlgorithm{Vector{Float64},Vector{Float64}}\n        stepsize::Float64\n        previous_q::Vector{Float64}\n        precision::Float64  # will hold the distance between the last two iterates\n        precisions::Vector{Float64} # record of all the precisions \n        SGDbis(stepsize::Float64) = new(stepsize, Vector{Float64}(), Inf, Vector{Float64}())\n    end\n    \n    # no changes\n    function (sgd::SGDbis)(problem::Any)\n        sgd.previous_q = rand(problem.n)\n    end\n    \n    # no changes\n    function (sgd::SGDbis)(q::Vector{Float64}, problem::Any)\n        sgd.stepsize * problem.∇f(q, rand(1:problem.m))\n    end\n    \n    function (sgd::SGDbis)(a::Vector{Float64}, worker::Int64, problem::Any) \n        q = sgd.previous_q - a \n        sgd.precision = norm(q-sgd.previous_q)\n        sgd.previous_q = q\n    end\nend","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"Recall that we defined const AIA = AsynchronousIterativeAlgorithms","category":"page"},{"location":"manual/#[stopnow](@ref)","page":"Manual","title":"stopnow","text":"","category":"section"},{"location":"manual/","page":"Manual","title":"Manual","text":"By default, you can specify the any of following stopping criteria through the stopat argument: maximum iteration, epoch and time. If any is met, the execution is stopped.","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"If you require additional stopping conditions, for instance \"stop at current iteration if the precision is below a threshold\" you can define stopnow on your algorithm:","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"function AIA.stopnow(sgd::SGDbis, stopat::NamedTuple) \n    haskey(stopat, :precision) ? sgd.precision ≤ stopat.precision : false\nend","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"You can now set stopat to (iteration=1000, precision=1e-5) or (precision=1e-5,) for example.","category":"page"},{"location":"manual/#[savenow](@ref)","page":"Manual","title":"savenow","text":"","category":"section"},{"location":"manual/","page":"Manual","title":"Manual","text":"By default, you can specify intervals at which some parameters are recorded through the saveat keyword: every iteration and every epoch.","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"If you require additional saving checkpoints, for instance \"save current iteration if is below a threshold\", you can define savenow on your algorithm:","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"function AIA.savenow(sgd::SGDbis, saveat::NamedTuple) \n    haskey(saveat, :precision) ? sgd.precision ≤ saveat.precision : false \nend","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"You can now set saveat to (precision=1e-4, time=42) or just (precision=1e-4,) for example.","category":"page"},{"location":"manual/#[savevalues](@ref)","page":"Manual","title":"savevalues","text":"","category":"section"},{"location":"manual/","page":"Manual","title":"Manual","text":"By default, at each saveat checkpoint, only queries, iterations, epochs, times, answer count per worker and optionally answers and their provenance.","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"If you want to record other values, for instance the precisions computed at the saveat checkpoints, you can define savevalues on your algorithm:","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"function AIA.savevalues(sgd::SGDbis) \n    sgd.precisions = append!(sgd.precisions, [sgd.precision])\nend","category":"page"},{"location":"manual/#[report](@ref)","page":"Manual","title":"report","text":"","category":"section"},{"location":"manual/","page":"Manual","title":"Manual","text":"To retrieve any values held by your algorithm, for example the precisions, return them as a NamedTuple in report:","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"function AIA.report(sgd::SGDbis)\n    (precisions = sgd.precisions,)\nend","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"They will now be outputted by start.","category":"page"},{"location":"manual/#[progress](@ref)","page":"Manual","title":"progress","text":"","category":"section"},{"location":"manual/","page":"Manual","title":"Manual","text":"When verbose>1 a progress bar is displayed. To reflect any progress other than the number of iterations, epochs, and the time, return a value between 0 to 1 (1 meaning completion) in progress:","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"function AIA.progress(sgd::SGDbis, stopat::NamedTuple) \n    if haskey(stopat, :precision) \n        sgd.precision == 0 && return 1.\n        return stopat.precision / sgd.precision\n    else \n        return 0.\n    end\nend","category":"page"},{"location":"manual/#[showvalues](@ref)","page":"Manual","title":"showvalues","text":"","category":"section"},{"location":"manual/","page":"Manual","title":"Manual","text":"Below the progress bar, by default, the number of iterations, epochs and answers-per-worker count are displayed. If you want to keep track of other values, return them as a Vector of Tuple{Symbol,Any} in savevalues:","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"function AIA.showvalues(sgd::SGDbis)\n    [(:precision, round(sgd.precision; sigdigits=4))]\nend","category":"page"},{"location":"manual/#Handling-worker-failures","page":"Manual","title":"Handling worker failures","text":"","category":"section"},{"location":"manual/","page":"Manual","title":"Manual","text":"If you expect some workers to fail but still want the algorithm to continue running, you can set the resilience parameter to the maximum number of worker failures you can tolerate before the execution is terminated.","category":"page"},{"location":"manual/#algorithm_wrappers","page":"Manual","title":"Algorithm wrappers","text":"","category":"section"},{"location":"manual/","page":"Manual","title":"Manual","text":"You are free to create your own algorithms, but if you're interested in aggregation algorithms, you can use an implementation provided in this library. The iteration of such an algorithm performs the following computation:","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"q_j longleftarrow textrmquery(underseti in textrmconnectedtextrmaggregate(a_j))  textrmwhere   a_i = textrmanswer(q_i)","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"where q_j is computed by the worker upon reception of textrmanswer(q_i) from worker j and where connected are the list of workers that have answered.","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"The AggregationAlgorithm in this library requires you to specify three methods: query, answer, and aggregate. Here's an example showing the required signatures of these three methods:","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"@everywhere begin \n    using Statistics\n\n    struct ToBeAggregatedGD <: AbstractAlgorithm{Vector{Float64},Vector{Float64}}\n        q1::Vector{Float64}\n        stepsize::Float64 \n    end\n\n    (tba::ToBeAggregatedGD)(problem::Any) = tba.q1\n    (tba::ToBeAggregatedGD)(a::Vector{Vector{Float64}}, connected::Vector{Int64}) = mean(a)            \n    (tba::ToBeAggregatedGD)(a::Vector{Float64}, problem::Any) = a\n    (tba::ToBeAggregatedGD)(q::Vector{Float64}, problem::Any) = q - tba.stepsize * problem.∇f(q)\nend \n\nalgorithm = AggregationAlgorithm(ToBeAggregatedGD(rand(10), 0.01); pids=workers())\n\nhistory = start(algorithm, distributed_problem, (epoch=100,));","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"Memory limitation: At any point in time, the central worker should have access must have access to the latest answers a_i from all the connected workers. This means storing a lot of a_i if we use many workers. There is a workaround when the aggregation operation is an average. In this case, only the equivalent of one answer needs to be saved on the central node, regardless of the number of workers.","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"AveragingAlgorithm implements this memory optimization. Here you only need to define query, the answer","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"@everywhere begin \n    struct ToBeAveragedGD <: AbstractAlgorithm{Vector{Float64},Vector{Float64}}\n        q1::Vector{Float64}\n        stepsize::Float64 \n    end\n\n    (tba::ToBeAveragedGD)(problem::Any) = tba.q1\n    (tba::ToBeAveragedGD)(a::Vector{Float64}, problem::Any) = a\n    (tba::ToBeAveragedGD)(q::Vector{Float64}, problem::Any) = q - tba.stepsize * problem.∇f(q)\nend \n\nalgorithm = AveragingAlgorithm(ToBeAveragedGD(rand(10), 0.01); pids=workers(), weights=ones(nworkers()))\n\nhistory = start(algorithm, distributed_problem, (epoch=100,));","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"Note that you can implement the custom callbacks on both these algorithms by defining them on your algorithm:","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"report(::ToBeAggregatedGD) = # do something","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"Hope you find this library helpful and look forward to seeing how you put it to use!","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = AsynchronousIterativeAlgorithms","category":"page"},{"location":"#[AsynchronousIterativeAlgorithms.jl](https://github.com/Selim78/AsynchronousIterativeAlgorithms.jl)","page":"Home","title":"AsynchronousIterativeAlgorithms.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"🧮 AsynchronousIterativeAlgorithms.jl handles the distributed asynchronous communications, so you can focus on designing your algorithm.","category":"page"},{"location":"","page":"Home","title":"Home","text":"💽 It also offers a convenient way to manage the distribution of your problem's data across multiple processes or remote machines.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"You can install AsynchronousIterativeAlgorithms by typing","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> ] add AsynchronousIterativeAlgorithms","category":"page"},{"location":"#quick_start","page":"Home","title":"Quick start","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Say you want to implement a distributed version of Stochastic Gradient Descent (SGD). You'll need to define:","category":"page"},{"location":"","page":"Home","title":"Home","text":"an algorithm structure subtyping AbstractAlgorithm{Q,A}\nthe initialization step where you compute the first iteration \nthe worker step performed by the workers when they receive a query q::Q from the central node\nthe asynchronous central step performed by the central node when it receives an answer a::A from a worker","category":"page"},{"location":"","page":"Home","title":"Home","text":"(Image: Sequence Diagram)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Let's first of all set up our distributed environment.","category":"page"},{"location":"","page":"Home","title":"Home","text":"# Launch multiple processes (or remote machines)\nusing Distributed; addprocs(5)\n\n# Instantiate and precompile environment in all processes\n@everywhere (using Pkg; Pkg.activate(@__DIR__); Pkg.instantiate(); Pkg.precompile())\n\n# You can now use AsynchronousIterativeAlgorithms\n@everywhere (using AsynchronousIterativeAlgorithms; const AIA = AsynchronousIterativeAlgorithms)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Now to the implementation.","category":"page"},{"location":"","page":"Home","title":"Home","text":"# define on all processes\n@everywhere begin\n    # algorithm\n    mutable struct SGD<:AbstractAlgorithm{Vector{Float64},Vector{Float64}}\n        stepsize::Float64\n        previous_q::Vector{Float64} # previous query\n        SGD(stepsize::Float64) = new(stepsize, Float64[])\n    end\n\n    # initialisation step \n    function (sgd::SGD)(problem::Any)\n        sgd.previous_q = rand(problem.n)\n    end\n\n    # worker step\n    function (sgd::SGD)(q::Vector{Float64}, problem::Any) \n        sgd.stepsize * problem.∇f(q, rand(1:problem.m))\n    end\n\n    # asynchronous central step\n    function (sgd::SGD)(a::Vector{Float64}, worker::Int64, problem::Any) \n        sgd.previous_q -= a\n    end\nend","category":"page"},{"location":"","page":"Home","title":"Home","text":"Now let's test our algorithm on a linear regression problem with mean squared error loss (LRMSE). This problem must be compatible with your algorithm. In this example, it means providing attributes n and m (dimension of the regressor and number of points), and the method ∇f(x::Vector{Float64}, i::Int64) (gradient of the linear regression loss on the ith data point)","category":"page"},{"location":"","page":"Home","title":"Home","text":"@everywhere begin\n    struct LRMSE\n        A::Union{Matrix{Float64}, Nothing}\n        b::Union{Vector{Float64}, Nothing}\n        n::Int64\n        m::Int64\n        L::Float64 # Lipschitz constant of f\n        ∇f::Function\n    end\n\n    function LRMSE(A::Matrix{Float64}, b::Vector{Float64})\n        m, n = size(A)\n        L = maximum(A'*A)\n        ∇f(x) = A' * (A * x - b) / n\n        ∇f(x,i) = A[i,:] * (A[i,:]' * x - b[i])\n        LRMSE(A, b, n, m, L, ∇f)\n    end\nend","category":"page"},{"location":"","page":"Home","title":"Home","text":"We're almost ready to start the algorithm...","category":"page"},{"location":"","page":"Home","title":"Home","text":"# Provide the stopping criteria \nstopat = (iteration=1000, time=42.)\n\n# Instanciate your algorithm \nsgd = SGD(0.01)\n\n# Create a function that returns an instance of your problem for a given pid \nproblem_constructor = (pid) -> LRMSE(rand(42,10),rand(42))\n\n# And you can [start](@ref)!\nhistory = start(sgd, problem_constructor, stopat);","category":"page"},{"location":"documentation/#Documentation","page":"Documentation","title":"Documentation","text":"","category":"section"},{"location":"documentation/#start","page":"Documentation","title":"start","text":"","category":"section"},{"location":"documentation/","page":"Documentation","title":"Documentation","text":"start","category":"page"},{"location":"documentation/#AsynchronousIterativeAlgorithms.start","page":"Documentation","title":"AsynchronousIterativeAlgorithms.start","text":"start(algorithm, problem_constructor, stopat; kwargs...)\nstart(algorithm, distributed_problem, stopat; kwargs...)\n\nSolve the distributed problem returned by problem_constructor (or referenced by distributed_problem) using the algorithm until the stopat conditions are reached.\n\nArguments\n\nalgorithm::AbstractAlgorithm{Q,A}: subtypes AbstractAlgorithm{Q,A} and implementing its functor calls\nproblem_constructor::Function: for each pid in {pids ⋃ current pid}, process pid calling problem_constructor(pid) should return the process' assigned problem\ndistributed_problem::DistributedObject: for each pid in {pids ⋃ current pid}, distributed_problem should reference process pid's assigned problem on pid\nstopat::NamedTuple: you can specify any of the following\niteration::Int64: maximum number of iterations\nepoch::Int64: maximum number of epochs (an epoch passes when all workers have answered at least one time) \ntime::Float64: maximum elapsed time (in seconds) \nother custom stopping conditions that you have specified by implementing stopnow\n\nKeywords\n\nsaveat=NamedTuple(): when to record query iterates (::Q), iterations, epochs, timestamps (and other custom values specified by implementing progress). Specified with any of the following \niteration::Int64: save every iteration> 0\nepoch::Int64: , save every epoch> 0\nother custom saving conditions that you have specified by implementing savenow\nsave_answers=false: answer iterates (::A) along with the pids of the workers that sent them are recorder \npids=workers(): pids of the active workers, you can start a non-distributed (and necessarily synchronous) version of your algorithm with pids=[1]\nsynchronous=false: if synchronous=true, the central node waits for all workers to answer before making a step\nresilience=0: number of workers allowed to fail before the execution is stopped\nverbose=1: if > 0, a progress bar is displayed (implent progress and/or showvalues to customize the display)\n\nReturns\n\nNamedTuple: a record of the queries and the iterations, epochs, timestamps at which they were recorded, as well an answer_count of each worker, additionally, \nif save_answers=true, a record of the answers and the answers_origin\nother custom values you have specified by implementing report\n\nThrows\n\nArgumentError: if the arguments don't match the specifications.\n\n\n\n\n\n","category":"function"},{"location":"documentation/#AbstractAlgorithm","page":"Documentation","title":"AbstractAlgorithm","text":"","category":"section"},{"location":"documentation/","page":"Documentation","title":"Documentation","text":"The algorithm you pass to start should subtype AbstractAlgorithm{Q,A}.","category":"page"},{"location":"documentation/","page":"Documentation","title":"Documentation","text":"AbstractAlgorithm","category":"page"},{"location":"documentation/#AsynchronousIterativeAlgorithms.AbstractAlgorithm","page":"Documentation","title":"AsynchronousIterativeAlgorithms.AbstractAlgorithm","text":"AbstractAlgorithm{Q,A}\n\nTo be compatible with start, types subtyping AbstractAlgorithm should be callable with the following signatures:\n\n(algorithm::AbstractAlgorithm{Q,A})(problem::Any)::Q where {Q,A}: the initialization step that create the first query iterate\n(algorithm::AbstractAlgorithm{Q,A})(q::Q, problem::Any)::A where {Q,A}: the answer step perfromed by the wokers when they receive a query q::Q from the central node\n(algorithm::AbstractAlgorithm{Q,A})(a::A, worker::Int64, problem::Any)::Q where {Q,A}: the query step performed by the central node when receiving an answer a::A from a worker\nwhen start takes the keyword synchronous=true, (algorithm::AbstractAlgorithm{Q,A})(as::Vector{A}, workers::Vector{Int64}, problem::Any)::Q where {Q,A}: the query step performed by the central node when receiving the answers as::Vector{A} respectively from the workers\n\n\n\n\n\n","category":"type"},{"location":"documentation/#Customization-of-start's-execution","page":"Documentation","title":"Customization of start's execution","text":"","category":"section"},{"location":"documentation/","page":"Documentation","title":"Documentation","text":"stopnow","category":"page"},{"location":"documentation/#AsynchronousIterativeAlgorithms.stopnow","page":"Documentation","title":"AsynchronousIterativeAlgorithms.stopnow","text":"stopnow(::AbstractAlgorithm, stopat::NamedTuple) = false\n\nDefine this method on your algorithm<:AbstractAlgorithm to add a stopping criterion:  return true if your stopping condition has been reached.\n\n\n\n\n\n","category":"function"},{"location":"documentation/","page":"Documentation","title":"Documentation","text":"savenow","category":"page"},{"location":"documentation/#AsynchronousIterativeAlgorithms.savenow","page":"Documentation","title":"AsynchronousIterativeAlgorithms.savenow","text":"savenow(::AbstractAlgorithm, saveat::NamedTuple) = false\n\nDefine this method on your algorithm<:AbstractAlgorithm to add saving stops:  return true if your saving condition has been reached\n\n\n\n\n\n","category":"function"},{"location":"documentation/","page":"Documentation","title":"Documentation","text":"savevalues","category":"page"},{"location":"documentation/#AsynchronousIterativeAlgorithms.savevalues","page":"Documentation","title":"AsynchronousIterativeAlgorithms.savevalues","text":"savevalues(::AbstractAlgorithm) = nothing\n\nDefine this method on your algorithm<:AbstractAlgorithm. It will be called at each iteration where savenow returns true:  store some values on your algorithm object (don't forget to define report to retrieve what you stored)\n\n\n\n\n\n","category":"function"},{"location":"documentation/","page":"Documentation","title":"Documentation","text":"report","category":"page"},{"location":"documentation/#AsynchronousIterativeAlgorithms.report","page":"Documentation","title":"AsynchronousIterativeAlgorithms.report","text":"report(::AbstractAlgorithm) = NamedTuple()\n\nDefine this method on your algorithm<:AbstractAlgorithm to add custom values to the results outputted by start:  return a NamedTuple() with those values, making sure to not reuse the field names queries, answers, iterations, epochs, timestamps, answers_origin, answer_count.\n\n\n\n\n\n","category":"function"},{"location":"documentation/","page":"Documentation","title":"Documentation","text":"progress","category":"page"},{"location":"documentation/#AsynchronousIterativeAlgorithms.progress","page":"Documentation","title":"AsynchronousIterativeAlgorithms.progress","text":"progress(::AbstractAlgorithm, stopat::NamedTuple) = 0.\n\nDefine this method on your algorithm<:AbstractAlgorithm to change the display of the progress bar:  return how close the current step is to reaching your stopping requirement on a scale of 0 to 1\n\n\n\n\n\n","category":"function"},{"location":"documentation/","page":"Documentation","title":"Documentation","text":"showvalues","category":"page"},{"location":"documentation/#AsynchronousIterativeAlgorithms.showvalues","page":"Documentation","title":"AsynchronousIterativeAlgorithms.showvalues","text":"showvalues(::AbstractAlgorithm) = Tuple{Symbol, Any}[]\n\nDefine this method on your algorithm<:AbstractAlgorithm to add a values to be displayed below the progress bar when verbose>1:  return a Tuple{Symbol, Any} with those values.\n\n\n\n\n\n","category":"function"},{"location":"documentation/#Algorithm-wrappers","page":"Documentation","title":"Algorithm wrappers","text":"","category":"section"},{"location":"documentation/","page":"Documentation","title":"Documentation","text":"The two following algorithms already subtype AbstractAlgorithm{Q,A} and are ready to use in start.","category":"page"},{"location":"documentation/","page":"Documentation","title":"Documentation","text":"AggregationAlgorithm","category":"page"},{"location":"documentation/#AsynchronousIterativeAlgorithms.AggregationAlgorithm","page":"Documentation","title":"AsynchronousIterativeAlgorithms.AggregationAlgorithm","text":"AggregationAlgorithm{Q,A,Alg<:AbstractAlgorithm{Q,A}}(arg; kwarg)::AbstractAlgorithm{Q,A} where {Q,A}\n\nDistributed algorithm that writes: q_j <- query(aggregate([answer(q_i) for i in connected])) Where a \"connected\" worker is a worker that has answered at least once. (Not memory optimized: length(pids) answers are stored on the central worker at all times)\n\nArgument\n\nalgorithm<:AbstractAlgorithm{Q,A} which should define the following\nalgorithm(problem::Any)::Q: the initialization step that create the first query iterate\nalgorithm(as::Vector{A}, workers::Vector{Int64})::AggregatedA where A: the answer aggregarion step performed by the central node when receiving the answers as::Vector{A} from the workers\nalgorithm(agg::AggregatedA, problem::Any)::Q: the query step producing a query from the aggregated answer agg::AggregatedA, performed by the central node\nalgorithm(q::Q, problem::Any)::A: the answer step perfromed by the wokers when they receive a query q::Q from the central node\n\nKeyword\n\npids=workers(): pids of the active workers\n\n\n\n\n\n","category":"type"},{"location":"documentation/","page":"Documentation","title":"Documentation","text":"AveragingAlgorithm","category":"page"},{"location":"documentation/#AsynchronousIterativeAlgorithms.AveragingAlgorithm","page":"Documentation","title":"AsynchronousIterativeAlgorithms.AveragingAlgorithm","text":"AveragingAlgorithm{Q,A,Alg<:AbstractAlgorithm{Q,A}}(arg; kwarg)::AbstractAlgorithm{Q,A} where {Q,A}\n\nDistributed algorithm that writes: q_j <- query(weighted_average([answer(q_i) for i in connected])) Where a \"connected\" worker is a worker that has answered at least once. (Memory optimized: only the equivalent of one answer is stored on the central worker at all times)\n\nArgument\n\nalgorithm<:AbstractAlgorithm{Q,A} which should define the following\nalgorithm(problem::Any)::Q: the initialization step that create the first query iterate\nalgorithm(a::A, problem::Any)::Q: the query step producing a query from the averaged answer, performed by the central node\nalgorithm(q::Q, problem::Any)::A: the answer step perfromed by the wokers when they receive a query q::Q from the central node\n\nKeyword\n\npids=workers(): pids of the active workers\n\n\n\n\n\n","category":"type"}]
}