"""
🧮 AsynchronousIterativeAlgorithms.jl handles the distributed asynchronous communication, so you can focus on
designing your algorithm.

💽 It also offers a convenient way to manage the distribution of your problem's data across multiple processes or
remote machines.

📚 Full manual available at https://selim78.github.io/AsynchronousIterativeAlgorithms.jl/dev/
"""
module AsynchronousIterativeAlgorithms

    include("network.jl")
    include("abstract_algorithm.jl")
    include("start.jl")
    include("algorithm_templates.jl")

end