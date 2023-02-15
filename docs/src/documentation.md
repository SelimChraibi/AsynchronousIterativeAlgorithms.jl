# Documentation

## `start` and `start!`

```@docs
start
```

```@docs
start!
```

## `AbstractAlgorithm`

The algorithm you pass to [`start`](@ref) should subtype `AbstractAlgorithm{Q,A}`.

```@docs
AbstractAlgorithm
```

## Algorithm templates

The two following algorithms already subtype [`AbstractAlgorithm{Q,A}`](@ref) and are ready to use in [`start`](@ref).

```@docs
AggregationAlgorithm
```

```@docs
AveragingAlgorithm

```
