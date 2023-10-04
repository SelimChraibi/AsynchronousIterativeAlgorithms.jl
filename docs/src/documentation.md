# Documentation

## `start`

```@docs
start
```

## `AbstractAlgorithm`

The algorithm you pass to [`start`](@ref) should subtype `AbstractAlgorithm{Q,A}`.

```@docs
AbstractAlgorithm
```

## Customization of `start`'s execution

```@docs
stopnow
```

```@docs
savenow
```

```@docs
savevalues
```

```@docs
report
```

```@docs
progress
```

```@docs
showvalues
```

## Algorithm wrappers

The two following algorithms already subtype [`AbstractAlgorithm{Q,A}`](@ref) and are ready to use in [`start`](@ref).

```@autodocs
AggregationAlgorithm
```

```@autodocs
AveragingAlgorithm
```
