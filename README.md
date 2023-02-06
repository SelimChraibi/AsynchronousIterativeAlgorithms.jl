# AsynchronousIterativeAlgorithms.jl

[![Build Status](https://github.com/Selim78/AsynchronousIterativeAlgorithms.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Selim78/AsynchronousIterativeAlgorithms.jl/actions/workflows/CI.yml?query=branch%3Amain)


🎷`AsynchronousIterativeAlgorithms.jl` handles the distributed asynchronous communication, allowing you to focus on designing your algorithm.

💽 It also offers a convenient way to manage the distribution of your problem's data across multiple processes or remote machines.


## Installation

You can install `AsynchronousIterativeAlgorithms` by typing

```julia
julia> ] add AsynchronousIterativeAlgorithms
```

## Quick start

![sequence diagram](./sequence.png "sequence diagram")

```julia
# Launch multiple processes (or remote machines)
using Distributed; addprocs(5)

# Instantiate and precompile environment in all processes
@everywhere (using Pkg; Pkg.activate(@__DIR__); Pkg.instantiate(); Pkg.precompile())

# You can now use AsynchronousIterativeAlgorithms
@everywhere (using AsynchronousIterativeAlgorithms; const AIA = AsynchronousIterativeAlgorithms)
```

Say you want to implemtent a distributed version of Stochastic Gradient Descent. You'll need to define...

```julia
@everywhere begin
    # ... an algorithm structure subtyping AbstractAlgorithm{Q,A}
    mutable struct SGD<:AbstractAlgorithm{Vector{Float64},Vector{Float64}}
        stepsize::Float64
        previous_q::Vector{Float64} # previous query
        SGD(stepsize::Float64) = new(stepsize, Vector{Float64}())
    end

    # ... the initialisation step where you compute the first iterate 
    function (sgd::SGD)(problem::Any)
        sgd.previous_q = rand(problem.m)
    end

    # ... the step perfromed by the workers when they receive a query `q::Q` from the central node
    function (sgd::SGD)(q::Vector{Float64}, problem::Any) 
        sgd.stepsize * problem.∇f(q, rand(1:problem.n))
    end

    # ... and the asynchronous step performed by the central node when receiving an answer `a::A` from a `worker`
    function (sgd::SGD)(a::Vector{Float64}, worker::Int64, problem::Any) 
        sgd.previous_q -= a
    end
end
```

Now let's test our algorithm on a linear regression problem. This problem should be compatible with your algorithm. In this example it means providing attributes `n` and `m` (number of points and dimension of the regressor) and the method `∇f(x::Vector{Float64}, i::Int64)` (gradient of the linear regression loss on the ith data point)

```julia
@everywhere begin
    struct LRMSE
        A::Union{Matrix{Float64}, Nothing}
        b::Union{Vector{Float64}, Nothing}
        n::Int64
        m::Int64
        L::Float64
        ∇f::Function
    end

    function LRMSE(A::Matrix{Float64}, b::Vector{Float64})
        n, m = size(A)
        L = maximum(A'*A)
        ∇f(x,i) = A[i,:] * (A[i,:]' * x - b[i])
        LRMSE(A, b, n, m, L, ∇f)
    end
end
```

We're almost ready to run our algorithm...

```julia
# Provide the stopping criteria 
stopat = (1000,0,0.) # (iterations, epochs, time)

# Instanciate your algorithm 
sgd = SGD(0.01)

# Create a function that returns an instance of your problem for a given pid 
problem_constructor = (pid) -> LRMSE(rand(42,10),rand(42))

# And you can run!
history = run(sgd, problem_constructor, stopat);
```

## Manual
<!-- no toc -->
- [Working with a distributed problem](#working-with-a-distributed-problem)
- [Synchronous run](#synchronous-run)
- [Active processes](#active-processes)
- [Recording iterates](#recording-iterates)
- [Custom stopping criterion](#custom-stopping-criterion)
- [`run!`](#run)
- [Algorithm templates](#algorithm-templates)
  

### Working with a distributed problem

When instanciating your problems you might have three requirement:

- **Limiting comunication costs** and **avoiding duplicated memory**: therefore loading the problems directly on their correct processes is be preferable to loading them on the central node before sending each of them to their process
- **Persistant data**: You might want to reuse problems you've created for other experiments

Depending on your needs, you have three options to construct your problems:

|  | communication costs <br> & duplicated memory | single use problems |
|---|:---:|:---:|
| Option 1 |  | ❌ |
| Option 2 | ❌ |  |
| Option 3 |  |  |

Now suppose you have a `make_problem` function

```julia
# Here we could also be reading the `A` and `b` from a file for example
@everywhere function make_problem(pid)
    pid==1 && return nothing # for now let's give process 1 an empty problem
    return LRMSE(rand(pid,10),rand(pid)) # here the sample size n == pid, don't do this irl
end
```

Here are your 3 options:

```julia
# Option 1: Instantiate the problems remotely at each process
problem_constructor = (pid) -> return make_problem(pid)

# Option 2: Instantiate the problems on the central node and send them to the problems
problems = Dict(procs() .=> make_problem.(procs()));
problem_constructor = (pid) -> return problems[pid]

# Option 3: Create a `DistributedObject` that references a problem on each active process. 
@everywhere using DistributedObjects
distributed_problem = DistributedObject((pid) -> make_problem(pid), pids=procs())
```

> Option 3 uses [`DistributedObjects`](https://github.com/Selim78/DistributedObjects.jl). In a nutshell, a `DistributedObject` instance references at most one object per process, and you can access the object stored on the current process with `[]`

As we saw *Option 2* should be discarded if you're working with large data, however it does allows access to the **global problem**, which *Option 1* doesn't:

```julia
function LRMSE(problems::Dict)
    pids = [pid for pid in keys(problems) if pid ≠ 1]
    n = sum([problems[pid].n for pid in pids])
    m = problems[pids[1]].m
    L = sum([problems[pid].L for pid in pids])
    ∇f(x) = sum([problems[pid].∇f(x) * problems[pid].n for pid in pids]) / n
    return LRMSE(nothing,nothing,n,m,L,∇f)
end

problems[1] = LRMSE(problems);
# We have access to the global Lipschitz constant!
sgd = SGD(1/problems[1].L)
```

*Option 3* is the best of both worlds:

```julia
function LRMSE(d::DistributedObject)
    pids = [pid for pid in where(d) if pid ≠ 1]
    n = sum(fetch.([@spawnat pid d[].n for pid in pids]))
    m = fetch(@spawnat pids[1] d[].m)
    L = sum(fetch.([@spawnat pid d[].L for pid in pids]))
    ∇f(x) = sum(fetch.([@spawnat pid d[].∇f(x) * d[].n for pid in pids])) / n
    return LRMSE(nothing,nothing,n,m,L,∇f)
end

distributed_problem[] = LRMSE(distributed_problem);
# We also have access to the global Lipschitz constant!
sgd = SGD(1/distributed_problem[].L)
```

Note that you can pass `distributed_problem::DistributedObject` to `run` instead of `problem_constructor::Function`

```julia
history = run(sgd, distributed_problem, stopat);
```

### Synchronous run

If you want to run your algorithm synchronously you just have to define a new method with following signature...

```julia
@everywhere begin
    # Synchronous step performed by the central node when receiving an answer `x::A` from a worker
    (sgd::SGD)(xs::Vector{Vector{Float64}}, workers::Vector{Int64}, problem::Any) = sum(xs)
end
```

...and to add the `synchronous=true` keyword to `run`

```julia
history = run(sgd, distributed_problem, stopat; synchronous=true);
```

### Active processes

You can chose which processes are active with the `pid` keyword

```julia
history = run(sgd, problem_constructor, stopat; pids=[2,3,6]);
```

If `pids=[1]`, `run` will run a non-distributed (and necessarily synchronous) version of your algorithm.

```julia
history = run(sgd, (pid)->LRMSE(rand(42,10),rand(42)), stopat; pids=[1], synchronous=true);
```

### Recording iterates

The query`::Q` iterates sent by the central nodes are saved every `iterations` and `epochs` in `saveat=(iterations, epochs)`

```julia
history = run(sgd, distributed_problem, stopat; saveat=(10,0));
```

You can also save the answers`::A` made by the workers by adding the `save_answers=true` keyword

```julia
history = run(sgd, distributed_problem, stopat; saveat=(10,0), save_answers=true);
```

### Custom stopping criterion

To **add** a stopping criterion to the default `stopat=(iteration, epoch, time)` you just need to

- define a new method `AsynchronousIterativeAlgorithms.stopnow` which will be dispatched when called on your algorithm
- declare that your algorithm implements the `Stoppable` trait

Let's modify the `SGD` example a bit to add a *precision* criterion

```julia
@everywhere begin
    using LinearAlgebra

    mutable struct CustomSGD<:AbstractAlgorithm{Vector{Float64},Vector{Float64}}
        stepsize::Float64
        previous_q::Vector{Float64}
        gap::Float64
        precision::Float64
        CustomSGD(stepsize::Float64, precision) = new(stepsize, Vector{Float64}(), 10^6, precision)
    end

    function (sgd::CustomSGD)(problem::Any) 
        sgd.previous_q = rand(problem.m)
    end

    function (sgd::CustomSGD)(q::Vector{Float64}, problem::Any)
        sgd.stepsize * problem.∇f(q, rand(1:problem.n))
    end

    function (sgd::CustomSGD)(a::Vector{Float64}, worker::Int64, problem::Any) 
        q = sgd.previous_q - a 
        sgd.gap = norm(q-sgd.previous_q)
        sgd.previous_q = q
    end

    # Stop when gap is small enough
    AIA.stopnow(sgd::CustomSGD) = sgd.gap ≤ sgd.precision
    AIA.Stoppability(::CustomSGD) = Stoppable()    
end

history = run(CustomSGD(0.01, 0.1), distributed_problem, (10,0,0.));
```

> This was only meant to be an example as in practice you can ask for a precision threshold by passing a fourth value in `stopat`. If you do so, and if you don't want to use the distance `(x,y)->norm(x-y)`, make sure to pass your distance function to the `distance` keyword of `run`.

```julia
history = run(CustomSGD(0.01, 0.1), distributed_problem, (10,0,0.,0.1));
```

### `run!`

By default, algorithms aren't modified by `run`. If you want it to be modified (to record some information during the optimization for instance) you can use `run!`.

### Algorithm templates

You are free to create your own algorithms. But if you're interested in *aggregation algorithms*, you might not have to implement everything from scratch. An iteration of an aggregation algorithm writes 
$$q_j \longleftarrow \textrm{query}(\underset{i \in \textrm{connected}}{\textrm{aggregate}}(a_j))\ \ \textrm{where }\ \ a_i = \textrm{answer}(q_i)$$  
where $q_j$ is computed by the worker uppon reception of $\textrm{answer}(q_i)$ from worker $j$ and where $connected$ are the list of worker that have answered.

An `AggregationAlgorithm` is already implemented in this library. To use it you need to specify the `query`, the `answer` and `aggregate`. Here's an example which shows the required signature of these three methods

```julia
@everywhere begin 
    using Statistics

    function agg_gd(q0, stepsize)
        initialize(problem::Any) = q0
        aggregate(a::Vector{Vector{Float64}}, connected::Vector{Int64}) = mean(a)            
        query(a::Vector{Float64}, problem::Any) = a
        answer(q::Vector{Float64}, problem::Any) = q - stepsize * problem.∇f(q)   

        AggregationAlgorithm{Vector{Float64}, Vector{Float64}}(initialize, aggregate, query, answer; pids=workers())
    end
end 

history = run(agg_gd(rand(10), 0.01), distributed_problem, (1000,0,0.));
```

**Memory limitation:** At any point in time, the central worker should have access to the most recent answers $a_i$ from *all* the connected workers. This means storing a lot of $a_i$ if we use many workers. There is a workaround when the aggregation operation is an *average*. In this case only the equivalent of one answer needs to be saved on the central node, no matter the number of workers.

`AveragingAlgorithm` implements this memory optimization. Here you only need to define `query`, the `answer`

```julia
@everywhere begin 
    # If you want the average to be weighted, you can add the keywords pids with their corresponding weights
    function avg_gd(q0, stepsize, pids=workers(), weights=ones(nworkers())) 
        initialize(problem::Any) = q0
        query(a::Vector{Float64}, problem::Any) = a
        answer(q::Vector{Float64}, problem::Any) =  q - stepsize * problem.∇f(q)
        AveragingAlgorithm{Vector{Float64}, Vector{Float64}}(initialize, query, answer; pids=pids, weights=weights)
    end
end

history = run(avg_gd(rand(10), 0.01), distributed_problem, (1000,0,0.));
```
