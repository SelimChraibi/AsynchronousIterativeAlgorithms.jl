# `AsynchronousIterativeAlgorithms.jl`

```@meta
CurrentModule = AsynchronousIterativeAlgorithms
```


```@docs
start(algorithm::AbstractAlgorithm{Q,A}, problem_constructor::Function, stopat::Union{Tuple{Int64, Int64, Float64}, Tuple{Int64, Int64, Float64, Float64}}; saveat=(0,0), save_answers=false, pids=workers(), synchronous=false, distance::Function=(x::Q,y::Q)->norm(x-y), resilience=0, verbose=1) where {Q,A}
```

```@docs
start(algorithm::AbstractAlgorithm{Q,A}, distributed_problem::DistributedObject{M}, stopat::Union{Tuple{Int64,Int64,Float64},Tuple{Int64,Int64,Float64,Float64}}; saveat=(0, 0), save_answers=false, pids=workers(), synchronous=false, distance::Function=(x::Q, y::Q) -> norm(x - y), resilience=0, verbose=1) where {Q,A,M}
```

```@docs
start!(algorithm::AbstractAlgorithm{Q,A}, problem_constructor::Function, stopat::Union{Tuple{Int64,Int64,Float64},Tuple{Int64,Int64,Float64,Float64}}; saveat=(0, 0), save_answers=false, pids=workers(), synchronous=false, distance::Function=(x::Q, y::Q) -> norm(x - y), resilience=0, verbose=1) where {Q,A}
```

```@docs
start!(algorithm::AbstractAlgorithm{Q,A}, distributed_problem::DistributedObject{M}, stopat::Union{Tuple{Int64,Int64,Float64},Tuple{Int64,Int64,Float64,Float64}}; saveat=(0, 0), save_answers=false, pids=workers(), synchronous=false, distance::Function=(x::Q, y::Q) -> norm(x - y), resilience=0, verbose=1) where {Q,A,M}
```
