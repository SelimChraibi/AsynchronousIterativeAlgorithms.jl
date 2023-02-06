using Test
using Distributed
addprocs(5)
@everywhere using Distributed
@everywhere using AsynchronousIterativeAlgorithms

@testset "AsynchronousIterativeAlgorithms.jl" begin

    ######################################
    #              network
    ######################################

    @everywhere using AsynchronousIterativeAlgorithms: Network, send_query, get_answer, send_answer, get_query, bind, close

    # Network sending and reveiving
    n = Network{Int64, Int64}()
    
    send_query(n, 111, 2)
    @test remotecall_fetch(get_query, 2, n) == 111
    remotecall_fetch(send_answer, 2, n, 42)
    @test get_answer(n) == (42 => 2)
    send_query(n, 123)
    @test remotecall_fetch(get_query, 2, n) == 123
    
    # binding a task to a network

    n = Network{Int64, Bool}()
    task = @task begin sleep(10) end
    bound_task = bind(n, task)
    schedule(bound_task)
    close(n)
    @test istaskdone(task)
    @test !n.isopen

    # binding a task to a network (terminating a finished task)

    n = Network{Int64, Bool}()
    task = @task begin sleep(0.01) end
    bound_task = bind(n, task)
    schedule(bound_task)
    wait(bound_task)
    close(n)
    @test istaskdone(task)
    
    # binding a task to a network (distributed version) 

    n = Network{Int64, Bool}()
    @everywhere 2 begin 
        task = @task begin sleep(10) end
        bound_task = bind($n, task)
        schedule(errormonitor(bound_task))
    end
    close(n)
    @test @everywhere 2 istaskdone(task)
end
