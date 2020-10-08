
using Distributions
using BenchmarkTools
#### Define Multiple Disbatch Quantile Function
function my_quantile(y::Float64,d::UnivariateDistribution)
    ### Initialization from Median as starting point
    x_0 = median(d)
    Xn = x_0
    ### Iterate through Newton's method
    for i in 1:20
        step = 1/pdf(d,x_0)
        Xn = Xn - step*(cdf(d,Xn)-y)
    end
    return Xn

end

#### Various Testing

############## Normal Distribution

d = Normal(0.0,1.0)

@btime my_quantile(0.25,d)
@btime quantile(d,0.25)

print("Gaussian Testing")
println()

print("Gaussian, 0.25, Newton")


println()
@btime print(my_quantile(0.25,d))
println()

print("Gaussian, 0.25, Library")
println()
print(quantile(d,0.25))
println()

print("Gaussian, 0.5, Newton")
println()
print(my_quantile(0.5,d))
println()
print("Gaussian, 0.5, Library")
println()

print(quantile(d,0.5))
println()

print("Gaussian, 0.75, Newton")
println()
print(my_quantile(0.75,d))
println()
print("Gaussian, 0.75, Library")
println()
print(quantile(d,0.75))
println()

################ Gamma Distribution

d = Gamma(5.0,1.0)

print("Gamma Testing")
println()

print("Gamma, 0.25, Newton")
println()
print(my_quantile(0.25,d))
println()

print("Gamma, 0.25, Library")
println()
print(quantile(d,0.25))
println()

print("Gamma, 0.5, Newton")
println()
print(my_quantile(0.5,d))
println()
print("Gamma, 0.5, Library")
println()

print(quantile(d,0.5))
println()

print("Gamma, 0.75, Newton")
println()
print(my_quantile(0.75,d))
println()
print("Gamma, 0.75, Library")
println()
print(quantile(d,0.75))
println()

################ Beta Distribution

d = Beta(2.0,4.0)

print("Beta, 0.25, Newton")
println()
print(my_quantile(0.25,d))
println()

print("Beta, 0.25, Library")
println()
print(quantile(d,0.25))
println()

print("Beta, 0.5, Newton")
println()
print(my_quantile(0.5,d))
println()
print("Beta, 0.5, Library")
println()
print(quantile(d,0.5))
println()

print("Beta, 0.75, Newton")
println()
print(my_quantile(0.75,d))
println()
print("Beta, 0.75, Library")
println()
print(quantile(d,0.75))

# All Results from this implementation of Newton's method
# are consistent with results obtained from Disibuted library
