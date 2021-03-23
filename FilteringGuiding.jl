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

struct Probability{T<:AbstractFloat}
    p::Array{T}
end

function backward(K::Kernel, h::HTransform)
    hout = HTransform(K.transitionmatrix * h.likelihood)
    Message(h, hout), hout
end

function changeofmeasure(K::Kernel, m::Message)#::typeof(K)  #change in place?
    Kstar = K.transitionmatrix .* transpose(m.h.likelihood)
    w = sum(Kstar, dims=2)                   #lots of flops; dont do
    Kstar = Kstar ./ replace!(w, 0=>1)       #changeofmeasure onto leaves!!
    #Kstar = Kstar ./ w                      #this div by 0 is very slow
    Kernel(Kstar)
end

function forward(Κ::Kernel, m::Message, μ::Probability)#::typeof(μ)?
    Kstar = changeofmeasure(K, m)
    μout = μ.p * Kstar.transitionmatrix
    Probability(μout)
end

function forward(Κ::Kernel, m::Message, x::Int)# sample, slow!
    Kstar = changeofmeasure(K, m)
    xout = Kstar.transitionmatrix[x,:]
    sample(Weights(xout))
end

elementwisemul(ha, hb) = ha.likelihood .* hb.likelihood

function fusion(branches)
    hout = reduce(elementwisemul, branches)
    HTransform(hout)
end
