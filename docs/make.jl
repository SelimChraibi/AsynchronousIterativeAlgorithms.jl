using Pkg; Pkg.activate("../"); Pkg.instantiate(); Pkg.precompile()
include("../src/AsynchronousIterativeAlgorithms.jl")

using Documenter
using .AsynchronousIterativeAlgorithms

@show AsynchronousIterativeAlgorithms

makedocs(
    sitename = "AsynchronousIterativeAlgorithms.jl",
    format = Documenter.HTML(),
    modules = [AsynchronousIterativeAlgorithms],
    pages = ["Home" => "index.md",
            "Manual" => "manual.md",
            "Documentation" => "documentation.md"]
)


deploydocs(
    repo = "github.com/selim78/AsynchronousIterativeAlgorithms.jl.git",
)
