using Distributed

# The central node queries the workers, the worker nodes answer
mutable struct Network{Q,A}
    pids::Vector{Int64}
    query_channels::Dict{Int64, RemoteChannel{Channel{Q}}}
    answer_channel::RemoteChannel{Channel{Pair{A, Int64}}}
    isopen::Bool
    interrupt_channel::RemoteChannel{Channel{Nothing}}
    bound_tasks_channel::RemoteChannel{Channel{Nothing}}
    check_interruption_channel::RemoteChannel{Channel{Nothing}}
    function Network{Q,A}(pids=workers()) where {Q,A}
        query_channels = Dict(worker => RemoteChannel(()->Channel{Q}(1), worker) for worker in pids);
        answer_channel = RemoteChannel(()->Channel{Pair{A, Int64}}(length(pids)))
        isopen = true
        interrupt_channel = RemoteChannel(()->Channel{Nothing}(1))
        bound_tasks_channel = RemoteChannel(()->Channel{Nothing}(length(pids)))
        check_interruption_channel = RemoteChannel(()->Channel{Nothing}(length(pids)))
        new(pids, query_channels, answer_channel, isopen, interrupt_channel, bound_tasks_channel, check_interruption_channel)
    end
end

# Do-block syntax
function Network{Q,A}(f::Function, pids=workers()) where {Q,A}
    network = Network{Q,A}(pids)
    try
        f(network)
    catch e 
        throw(e)
    finally
        close(network)
    end
end

# central node side
send_query(n::Network{Q,A}, q::Q, pid::Int64) where {Q,A} = (put!(n.query_channels[pid], q); n)
send_query(n::Network{Q,A}, q::Q) where {Q,A} = (foreach((pid) -> put!(n.query_channels[pid], q), n.pids); n)
get_answer(n::Network{Q,A}) where {Q,A} = take!(n.answer_channel)::Pair{A,Int64}

# worker nodes side
send_answer(n::Network{Q,A}, a::A) where {Q,A} = (put!(n.answer_channel, (a => myid())); n)
get_query(n::Network{Q,A}) where {Q,A} = take!(n.query_channels[myid()])::Q

"""
    bind(n::Newtork, t::Task)

Associate the lifetime of the task `t` with the network `n`: `t` is interrupted when `n` is closed. 
"""
function Base.bind(n::Network, t::Task)
    put!(n.bound_tasks_channel, nothing)
    @task try
        schedule(t)
        @async (wait(n.interrupt_channel); schedule(t, InterruptException(), error=true))
        wait(t)
    catch e 
        if isa(e, TaskFailedException) 
            !isa(e.task.exception, InterruptException) && throw(e.task.exception)
        else 
            throw(e)
        end
    finally
        put!(n.check_interruption_channel, nothing)
    end
end

"""
    close(n::Newtork)

Interrupt tasks feeding the network in the workers, then close all channels of the network. 
"""
function Base.close(n::Network)
    put!(n.interrupt_channel, nothing)
    while isready(n.bound_tasks_channel)
        take!(n.bound_tasks_channel) 
        take!(n.check_interruption_channel)
    end
    close.([n.answer_channel, values(n.query_channels)...])
    close.([n.interrupt_channel, n.bound_tasks_channel, n.check_interruption_channel])
    n.isopen = false;
end

Base.isopen(n::Network) = n.isopen

# Just for the convenience of broadcasting networks
Base.broadcastable(n::Network) = Ref(n)