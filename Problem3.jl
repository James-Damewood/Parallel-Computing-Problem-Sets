using Plots
using BenchmarkTools
using Distributed


@everywhere function f(x)
    return x*(1-x)
end


@everywhere function calc_attractor!(out,ind,f,p,num_attract=150;warmup=400)
    X_n = 0.25
    for i in 1:warmup
        X_n = p*f(X_n)
    end
    for i in 1:num_attract
        X_n = p*f(X_n)
        out[ind,i] = X_n
    end
    return nothing
end


################################ 3.1   Test Everything is working

ss_attract = Array{Float64}(undef, 1,150)

print("Timing Single Traj")
@btime calc_attractor!(ss_attract,1,f,2.9)
################ Timing is 1.727 us
println()

ss_attract[1,150]

print("Computed")
print(ss_attract[1,150])
println()
print("Analytical")
print(0.655172)


###### Comuted Value is 0.6551724137931038
###### Analytical Value is root of
###### x - 2.9x +2.9x^2 = 0
###### Root is 0.655172

##### These values are consistent

############################### 3.2   Testing different Parameters

const bf_data = Array{Float64}(undef, 1101,150)

function bifurcate_series(out,calc,f,r)
    for (index,val) in enumerate(r)
        calc(out,index,f,val)
    end
    return nothing
end

const r_range = LinRange(2.9,4.0,1101)

print("Timing Parameter Search Sequential")
@btime bifurcate_series(bf_data,calc_attractor!,f,r_range)
###### Timing is 1.809 ms
println()

plot(r_range, bf_data, seriestype = :scatter, legend=false,title = "Bifurcation")

######## See Plot output in HW_0 folder

############################### 3.3    MultiThreading

const cache_parallel = [Array{Float64}(undef,1,150) for i in 1:Threads.nthreads()]
function trajectory(f,r)
  # u is automatically captured
  calc_attractor!(cache_parallel[Threads.threadid()],1,f,r);
  return cache_parallel[Threads.threadid()]
end

trajectory(f,2.9)

cache_parallel

function tmap(t,rs)
  out = Array{Float64}(undef, 1101,150)
  Threads.@threads for i in 1:1101
    # each loop part is using a different part of the data
    #print(t(rs[i])[1])
    out[i,:,:] = t(rs[i])
  end
  return out
end

print("Time Parameter Search Multi-Threading")
@btime threaded_out = tmap(r -> trajectory(f,r),r_range)
####### Timing is 1.028 ms
threaded_out = tmap(r -> trajectory(f,r),r_range)
println()
####### passes accuracy test compared to serial
diff = bf_data - threaded_out[:,:]

################## Discussion

# I ran this using two threads.
# The multithreading version of the search ran
# approximately slightly over twice as fast as the sequential search.
# The computation is leveraging the multiple threads
# to compute the trajectories in parallel.
# These results are within the range of expectations.

################################### 3.4    MultiProcess

using SharedArrays
using Distributed
addprocs(2)
pmap_test = SharedArray{Float64,2}(1101,150)

distri_test = SharedArray{Float64,2}(1101,150)

@everywhere function f(x)
    return x*(1-x)
end

@everywhere function calc_attractor!(out,ind,f,p,num_attract=150;warmup=400)
    X_n = 0.25
    for i in 1:warmup
        X_n = p*f(X_n)
    end
    for i in 1:num_attract
        X_n = p*f(X_n)
        out[ind,i] = X_n
    end
    return nothing
end

@everywhere function pmap_calc_attractor!(out,f,p,num_attract=150;warmup=400)
    X_n = 0.25
    for i in 1:warmup
        X_n = p*f(X_n)
    end
    for i in 1:num_attract
        X_n = p*f(X_n)
        out[i] = X_n
    end
    out
end

r_values_pmap = Array(2.9:0.001:4)
###### Pmap
print("Time PMAP")
@btime pmap(i -> pmap_calc_attractor!(view(pmap_test,i,:),f,r_values_pmap[i]),1:length(r_values_pmap))
###### Timing is 1.258 s
pmap_test

###### Passes Accuracy Test compared to serial
diff = bf_data - pmap_test

###### Distributed

@everywhere function dis_operation(out,calc,f,r)
    @sync @distributed for i in 1:1101
        calc(out,i,f,r[i])
    end
end

print("Time Distributed")
@btime dis_operation(distri_test,calc_attractor!,f,r_range)
###### timing is 1.618ms
distri_test

####### Passes Accuracy Test compared to serial
diff = bf_data - distri_test


####################### 3.5 Discussion

#### PMAP is a far outlier in this test as it required
# on the order of 1s, while the multi-threaded and
# distributed methods performed on the order of 1ms.
# PMAP works as a dynamic scheduler, and because all of the tasks in this
# case are very similar, the overhead computation run by the dynamic
# scheduler is very inefficient.
# The mulithreaded (1.03ms) and the @distributed (1.6ms) are much closer.
# The @distributed model may be slower due to the overhead involved in the
# parallelization, such as the time required to transfer instructions and
# data between the master and the various workers.
# The similarity between all of the required processes suits the problem
# particularly well for multithreading.
