using Distributed

# The central node queries the workers, the worker nodes answer
mutable struct Network{Q,A}
    pids::Vector{Int64}
    query_channels::Dict{Int64,RemoteChannel{Channel{Q}}}
    answer_channel::RemoteChannel{Channel{Pair{A,Int64}}}
    isopen::Bool
    resilience::Int64
    error_channel::RemoteChannel{Channel{Pair{Int64,CapturedException}}}
    interrupt_channel::RemoteChannel{Channel{Nothing}}
    bound_tasks_channel::RemoteChannel{Channel{Nothing}}
    confirm_interruption_channel::RemoteChannel{Channel{Nothing}}
    function Network{Q,A}(pids=workers(), resilience=0) where {Q,A}
        query_channels = Dict(worker => RemoteChannel(() -> Channel{Q}(1), worker) for worker in pids)
        answer_channel = RemoteChannel(() -> Channel{Pair{A,Int64}}(length(pids)))
        isopen = true
        error_channel = RemoteChannel(() -> Channel{Pair{Int64,CapturedException}}(length(pids)))
        interrupt_channel = RemoteChannel(() -> Channel{Nothing}(1))
        bound_tasks_channel = RemoteChannel(() -> Channel{Nothing}(length(pids)))
        confirm_interruption_channel = RemoteChannel(() -> Channel{Nothing}(length(pids)))
        new(pids, query_channels, answer_channel, isopen, resilience, error_channel, interrupt_channel, bound_tasks_channel, confirm_interruption_channel)
    end
end

# central node side
send_query(n::Network{Q,A}, q::Q, pid::Int64) where {Q,A} = (put!(n.query_channels[pid], q); n)
send_query(n::Network{Q,A}, q::Q) where {Q,A} = (foreach((pid) -> put!(n.query_channels[pid], q), n.pids); n)
get_answer(n::Network{Q,A}) where {Q,A} = take!(n.answer_channel)::Pair{A,Int64}

# worker nodes side
send_answer(n::Network{Q,A}, a::A) where {Q,A} = (put!(n.answer_channel, (a => myid())); n)
get_query(n::Network{Q,A}) where {Q,A} = take!(n.query_channels[myid()])::Q

# Exception to be thrown to the central process when enough workers error
struct WorkerFailedException <: Exception
    error_count::Int64
    error::RemoteException
end
function Base.showerror(io::IO, e::WorkerFailedException)
    msg = (e.error_count == 0 ? "A worker failed:" : "Maximum number of worker failure authorized ($(e.error_count)) was reached:")
    println(io, msg)
    showerror(io, e.error)
end

# Interrupt the network `n` when `n.resilience` errors have been thrown by workers
function central_interruptor(n::Network, task::Task)
    @async begin
        error_count = 0
        # print the errors below the threshold
        while error_count < n.resilience
            worker, e = take!(n.error_channel)
            showerror(stdout, RemoteException(worker, e))
            error_count += 1
        end

        # interrupt when the threshold is reached
        worker, e = take!(n.error_channel)
        we = WorkerFailedException(error_count, RemoteException(worker, e))
        schedule(task, we, error=true)
    end
end

# Execute `f(network)` and interrupt it when `resilience` processes fail
function open_network(f::Function, Q::Type, A::Type, pids=workers(), resilience=0)
    network = Network{Q,A}(pids, resilience)
    task = @task f(network)

    interruptor = central_interruptor(network, task)

    try
        schedule(task)
        wait(task)
    catch e
        if !isa(e, TaskFailedException)
        else
            rethrow(e.task.exception)
        end
    finally
        # interrupt the interruptor
        try
            schedule(interruptor, InterruptException(), error=true)
        catch
        end
        close(network)
    end
end

# Exception to be thrown at tasks bound to the network upon closure
struct NetworkClosingException <: Exception end
Base.showerror(io::IO, e::NetworkClosingException) = println(io, "Network is closing.")

# interrupt the task when the network puts in its `interrupt_channel`
function worker_interruptor(t::Task, n::Network)
    # register a new bound task to the network 
    put!(n.bound_tasks_channel, nothing)

    @async try
        wait(n.interrupt_channel)
        schedule(t, NetworkClosingException(), error=true)
        wait(t)
    finally
        put!(n.confirm_interruption_channel, nothing)
    end
end

# executes f and interrupts it when the `network` is closed
function bound_to(f::Function, network::Network)
    task = @task f(network)
    interruptor = worker_interruptor(task, network)
    try
        schedule(task)
        wait(task)
    catch e
        if !(isa(e, TaskFailedException) && isa(e.task.exception, NetworkClosingException))
            put!(network.error_channel, (myid() => CapturedException(task.exception, task.backtrace)))
        end
    finally
        try
            schedule(interruptor, InterruptException(), error=true)
        catch
        end
    end
end

"""
    close(n::Newtork)

Interrupt worker tasks feeding the network, then close all channels of the network. 
"""
function Base.close(n::Network)
    put!(n.interrupt_channel, nothing)
    while isready(n.bound_tasks_channel)
        take!(n.bound_tasks_channel)
        take!(n.confirm_interruption_channel)
    end
    close.([n.answer_channel, values(n.query_channels)...])
    close.([n.interrupt_channel, n.bound_tasks_channel, n.confirm_interruption_channel])
    n.isopen = false
    nothing
end

Base.isopen(n::Network) = n.isopen