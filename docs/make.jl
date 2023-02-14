using Documenter
using AsynchronousIterativeAlgorithms

makedocs(
    sitename = "AsynchronousIterativeAlgorithms.jl",
    authors = "Selim Chraibi"
    format = Documenter.HTML(),
    modules = [AsynchronousIterativeAlgorithms],
    pages = ["Home" => "index.md",
            "Manual" => "manual.md",
            "Documentation" => "documentation.md"]
)

deploydocs(
    repo = "github.com/Selim78/AsynchronousIterativeAlgorithms.jl",
)
