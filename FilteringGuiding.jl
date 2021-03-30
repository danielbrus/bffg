using LinearAlgebra, StatsBase

struct HTransform{T<:AbstractFloat}
    likelihood::Vector{T}
end

struct ProbabilityDistribution{T<:AbstractFloat}
    p::Vector{T}
end

struct Message{T<:AbstractFloat}
    h::HTransform{T}
    honto::HTransform{T}
end

struct Kernel{T<:AbstractFloat}
    transitionmatrix::Matrix{T}
end

Base.:*(K::Kernel, h::HTransform) = HTransform(K.transitionmatrix * h.likelihood)

Base.:*(ha::HTransform, hb::HTransform) = HTransform(ha.likelihood .* hb.likelihood)

LinearAlgebra.:dot(P::ProbabilityDistribution, h::HTransform) = dot(P.p, h.likelihood)

Base.getindex(h::HTransform, i::Int) = h.likelihood[i]

function backward(K::Kernel, h::HTransform)
    hout = K * h
    Message(h, hout), hout
end

function forward(P::ProbabilityDistribution, m::Message, x::Integer) #, win::AbstractFloat)
    v = changeofmeasure(P, m.h)
    wnew = dot(P, m.h) / m.honto[x]                     # weight
    sample(eachindex(v), weights(v)), log(wnew) #+win   # (sample, wnew)
end

function changeofmeasure(P::ProbabilityDistribution, h::HTransform)
    P.p .* h.likelihood                   # do we need this function
end

function fusion(branches::Array{HTransform{AbstractFloat}})
    reduce(*, branches)
end

#=
function forward(K::Kernel, m::Message, x::Integer, ω::AbstractFloat)
    v = changeofmeasure(K, m.h)
    w = (K * m.h)[1] / m.honto[x]         # K should be p-dist
    sample(eachindex(v), weights(v)), ω*w # v unnormalised and potentially zeros
end

function changeofmeasure(K::Kernel, h::HTransform)  # change in place?
    K.transitionmatrix .* transpose(h.likelihood)   # should act on p-dist
end

function forward(K::Kernel, m::Message)        # state is used in Kernel construction
    Kstar = changeofmeasure(K, m)              # no slicing/viewing neccesary
    v = Kstar.transitionmatrix  # [x,:]         try except sum(v)==0
    sample(Weights(vec(v)))                    # slower on sum(v)!=1 but faster than v/sum(v)
end

#Base.:*(P::ProbabilityDistribution, h::HTransform) = dot(P.p, h.likelihood)
=#
