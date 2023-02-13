push!(LOAD_PATH,"../src/")
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

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
