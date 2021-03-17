using BenchmarkTools
using StatsBase

struct HTransform{T}
    Likelihood::Vector{T}
end

struct Message
    h::HTransform
    #honto::HTransform
end

struct Kernel{T<:AbstractFloat}
    TransitionMatrix::Matrix{T}
end

struct Probability{T<:Number}
    p::Array{T}
    #function Probability(a::Array{T}) where T
    #    new{T}(a/sum(a))
    #end
end

function backward(Κ::Kernel, h::HTransform) #specify return :: type ?
    hout = HTransform(K.TransitionMatrix*h.Likelihood)
    Message(h), hout
end

function changeofmeasure(K::Kernel, m::Message)
    K_star = K.TransitionMatrix .* transpose(m.h.Likelihood)
    w = sum(K_star, dims=2)
    K_star = K_star ./ replace!(w, 0=>1)
    Kernel(K_star)
end

function forward(Κ::Kernel, m::Message, μ::Probability)
    ## Julia best practice: Should i Initialise a new K_star = Kernel(K.Trans....) ?
    #=
    K_star = Κ.TransitionMatrix .* transpose(m.h.Likelihood)
    w = sum(K_star, dims=2)
    K_star = K_star ./ replace!(w, 0=>1)   # if row sums to 0 dont div by 0
    =#
    μout = Probability(μ.p * changeofmeasure(K, m).TransitionMatrix)

    #μout = Probability(μ.p * K_star)
    μout
end

#=
function forward(Κ::Kernel, m::Message, x::Integer)
    ## Julia best practice: Should i Initialise a new K_star = Kernel(K.Trans....) ?
    K_star = Κ.TransitionMatrix .* transpose(m.h.Likelihood)
    ## this whole function is kind of 'unfunctional' not sure if this is inline with paradigm
    w = sum(K_star, dims=2)
    K_star = K_star ./ replace!(w, 0=>1)   # if row sums to 0 dont div by 0
    xout = sample(Weights(K_star[x,:]))
    xout
end
=#

elementwise(ha::HTransform, hb::HTransform) = ha.Likelihood .* hb.Likelihood

function fusion(branches)
    hout = reduce(elementwise, branches)   #or HTransform(reduce ... ) ?
    HTransform(hout)
end

K = Kernel([3/4    1/4  0
            0      1/2  1/2
            1/10   0    9/10])

#=
@btime begin
    h4a = HTransform([0.0, 0.0, 1.0]) #x4a = '3'
    m4a, h34a = backward(K, h4a)
end
=#

h4a = HTransform([0.0, 0.0, 1.0]) #x4a = '3'
m4a, h34a = backward(K, h4a)

h4b = HTransform([0.0, 1.0, 0.0]) #x4b = '2'
m4b, h34b = backward(K, h4b)

#=
@btime h3 = fusion([h34a, h34b])
=#

h3 = fusion([h34a, h34b])

m3, h2 = backward(K, h3)
m2, h1 = backward(K, h2)
m1, h0 = backward(K, h1)

μ0 = Probability([1/3 1/3 1/3])

#μ0 = 1

#=
@btime μ1 = forward(K, m1, μ0)
=#

μ1 = forward(K, m1, μ0)
μ2 = forward(K, m2, μ1)
μ3 = forward(K, m3, μ2)

μ4a = forward(K, m4a, μ3)
μ4b = forward(K, m4b, μ3)

println(μ0)
println(μ1)
println(μ2)
println(μ3)
println(μ4a)
println(μ4b)
