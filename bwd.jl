#h3 = Array{Float64,1}(undef, 3)  # pre allocate everything?

#=
function Backward!(Κ, h, hout)  # avoid allocations?
    hout = K*h
end
=#
using StatsBase

struct hTransform
    Likelihood::Vector
end

struct Message
    h::hTransform
    #honto::hTransform
end

struct Kernel
    TransitionMatrix::Matrix
end

struct Probability
    Distribution::Array
end

function Backward(Κ::Kernel, h::hTransform) #specify return :: type ?
    hout = hTransform(K.TransitionMatrix*h.Likelihood)
    Message(h), hout
end

function Forward(Κ::Kernel, m::Message, μ::Probability)
    ## Julia best practice: Should i Initialise a new K_star = Kernel(K.Trans....) ?
    K_star = Κ.TransitionMatrix .* transpose(m.h.Likelihood)
    w = sum(K_star, dims=2)
    K_star = K_star ./ replace!(w, 0=>1)   # if row sums to 0 dont div by 0
    μout = Probability(μ.Distribution * K_star)
    μout
end

function Forward(Κ::Kernel, m::Message, x::Int64)
    ## Julia best practice: Should i Initialise a new K_star = Kernel(K.Trans....) ?
    K_star = Κ.TransitionMatrix .* transpose(m.h.Likelihood)
    ## this whole function is kind of 'unfunctional' not sure if this is inline with paradigm
    w = sum(K_star, dims=2)
    K_star = K_star ./ replace!(w, 0=>1)   # if row sums to 0 dont div by 0
    xout = sample(Weights(K_star[x,:]))
    xout
end

elementwise(ha::hTransform, hb::hTransform) = ha.Likelihood .* hb.Likelihood

function Fusion(branches)
    hout = reduce(elementwise, branches)   #or hTransform(reduce ... ) ?
    hTransform(hout)
end

K = Kernel([3/4    1/4  0
            0      1/2  1/2
            1/10   0    9/10])

h4a = hTransform([0, 0, 1]) #x4a = '3'
h4b = hTransform([0, 1, 0]) #x4b = '2'

m4a, h34a = Backward(K, h4a)
m4b, h34b = Backward(K, h4b)

h3 = Fusion([h34a, h34b])

m3, h2 = Backward(K, h3)
m2, h1 = Backward(K, h2)
m1, h0 = Backward(K, h1)

μ0 = Probability([1/3 1/3 1/3])
#μ0 = 1

μ1 = Forward(K, m1, μ0)
μ2 = Forward(K, m2, μ1)
μ3 = Forward(K, m3, μ2)

μ4a = Forward(K, m4a, μ3)
μ4b = Forward(K, m4b, μ3)
