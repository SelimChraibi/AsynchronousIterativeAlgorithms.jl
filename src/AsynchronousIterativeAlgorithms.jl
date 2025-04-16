"""
AsynchronousIterativeAlgorithms.jl handles the distributed asynchronous communication, so you can focus on
designing your algorithm.

It also offers a convenient way to manage the distribution of your problem's data across multiple processes or
remote machines.

A full manual is available at https://SelimChraibi.github.io/AsynchronousIterativeAlgorithms.jl/dev/
"""
module AsynchronousIterativeAlgorithms

    include("network.jl")
    include("abstract_algorithm.jl")
    include("start.jl")
    include("algorithm_wrappers.jl")

    export AbstractAlgorithm, 
           stopnow, 
           savenow, 
           savevalues, 
           report, 
           progress, 
           showvalues, 
           start, 
           start!
end
