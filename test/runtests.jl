using Test
using Distributed
addprocs(5)
@everywhere using Distributed
@everywhere using AsynchronousIterativeAlgorithms

@testset "AsynchronousIterativeAlgorithms.jl" begin

    ######################################
    #              network
    ######################################

    @everywhere using AsynchronousIterativeAlgorithms: Network, send_query, get_answer, send_answer, get_query, open_network, bound_to, close, WorkerFailedException

    # Network sending and reveiving
    n = Network{Int64, Int64}()
    
    send_query(n, 111, 2)
    @test remotecall_fetch(get_query, 2, n) == 111
    remotecall_fetch(send_answer, 2, n, 42)
    @test get_answer(n) == (42 => 2)
    send_query(n, 123)
    @test remotecall_fetch(get_query, 2, n) == 123

    close(n)
    @test !isopen(n)
    
    # Executions bound to the network
    @everywhere function test_send_receive(network)
        bound_to(network) do network
            q = get_query(network)
            send_answer(network, q*"answer")
        end
    end
    
    open_network(String, String) do network
        @async remotecall_fetch(test_send_receive, 2, network)
        send_query(network, "query", 2)
        a, worker = get_answer(network)
        @test a == "queryanswer" && worker == 2
    end

    # Error at worker
    @everywhere function test_error_at_worker(network)
        bound_to(network) do network
            error()
        end
    end
    
    @test try
        open_network() do network
            remotecall_fetch(test_error_at_worker, 2, network)
        end
        false
    catch e
        true
    end

    # Error at worker with resilience
    @everywhere function test_error_at_worker(network)
        bound_to(network) do network
            error("This error should be printed")
        end
    end
    
    @test try
        open_network(; resilience=1) do network
            remotecall_fetch(test_error_at_worker, 2, network)
        end
        true
    catch e
        false
    end


    # Bound worker execution dies when central ends
    rc = RemoteChannel(()->Channel(1))

    @everywhere function test_end_at_central(network, rc)
        bound_to(network) do network
            sleep(0.1)
            put!(rc, 42)
        end
    end

    open_network() do network
        @async remotecall_fetch(test_end_at_central, 2, network, rc)
    end

    @test !isready(rc)

    # sanity check: we wait for test_error_at_central
    open_network() do network
        remotecall_fetch(test_end_at_central, 2, network, rc)
    end

    @test isready(rc)

end