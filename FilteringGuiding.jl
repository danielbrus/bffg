import StatsBase.sample, StatsBase.Weights

struct HTransform{T<:AbstractFloat}
    likelihood::Vector{T}
end

struct Message{T<:AbstractFloat}
    h::HTransform{T}
    honto::HTransform{T}
end

struct Kernel{T<:AbstractFloat}
    transitionmatrix::Matrix{T}
end

function backward(K::Kernel, h::HTransform)
    hout = HTransform(K.transitionmatrix * h.likelihood)
    Message(h, hout), hout
end

function forward(K::Kernel, m::Message, x::Integer)# state is used in Kernel construction
    Kstar = changeofmeasure(K, m)                # no slicing/viewing neccesary
    sample(Weights(vec(Kstar.transitionmatrix)))
end

function changeofmeasure(K::Kernel, m::Message)  # change in place?
    Kstar = K.transitionmatrix .* transpose(m.h.likelihood)
    w = sum(Kstar, dims=2)                       # computationally intensive
    Kstar = Kstar ./ replace!(w, 0=>1)           # not sure how to fix
    #Kstar = Kstar ./ w                          # (need to protect against ÷ 0)
    Kernel(Kstar)
end

function fusion(branches)
    hout = reduce(elementwisemul, branches)
    HTransform(hout)
end

elementwisemul(ha, hb) = ha.likelihood .* hb.likelihood
