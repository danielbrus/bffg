include("FilteringGuiding.jl")
import Plots.heatmap

function ψ(u::AbstractFloat)
    τ = 0.1
    exp(-τ * u)
end

function Kappa(x::Array{Int64,1}, i::Integer)
    δ = 0.001
    μ = 0.6
    ν = 0.1
    λ = 2.5

    neighbours = [j for j in [i-2, i-1, i+1, i+2] if j > 0 && j <= length(x)]
    N = count(map(n -> n == 2, x[neighbours]))   # neighbour == 2 (:= 'infected')

    if x[i] == 1
        K = [(1. - δ) * ψ(λ * N)    1. - (1. - δ) * ψ(λ * N)    0.       ]
    end
    if x[i] == 2
        K = [0.                     ψ(μ)                        1. - ψ(μ)]
    end
    if x[i] == 3
        K = [1. - ψ(ν)              0.                          ψ(ν)     ]
    end
    Kernel(K)
end

function Kappa()
    δ = 0.001
    μ = 0.6
    ν = 0.1
    Λ = 0.5

    Kernel([(1. - δ) * ψ(Λ)    1. - (1. - δ) * ψ(Λ)    0.
            0.                 ψ(μ)                    1. - ψ(μ)
            1. - ψ(ν)          0.                      ψ(ν)     ])
end

function backward(K::Kernel, hs::Array{HTransform{Float64},1})
    N = length(hs)
    ms = Array{Message{Float64}}(undef, N)
    houts = Array{HTransform{Float64}}(undef, N)
    for i = 1:N
        ms[i], houts[i] = backward(K, hs[i])
    end
    ms, houts
end

function forward(K::Function, ms::Array{Message{Float64},1}, x::Array{Int64,1})
    N = length(x)
    xnext = Array{Int64,1}(undef, N)
    for i = 1:N
        xnext[i] = forward(K(x, i), ms[i], x[i])
    end
    xnext
end

Ktilde = Kappa()

h4a = HTransform([0., 0., 1.]) # 'R'
h4b = HTransform([0., 0., 1.]) # 'R'
h4c = HTransform([0., 1., 0.]) # 'I'
h4d = HTransform([0., 0., 1.]) # 'R'
h4e = HTransform([0., 0., 1.]) # 'R'

h4 = [h4a, h4b, h4c, h4d, h4e] # final observations

m4, h3 = backward(Ktilde, h4)
m3, h2 = backward(Ktilde, h3)
m2, h1 = backward(Ktilde, h2)
m1, h0 = backward(Ktilde, h1)

x0 = [2, 1, 1, 1, 1]           # initial observations

x1 = forward(Kappa, m1, x0)
x2 = forward(Kappa, m2, x1)
x3 = forward(Kappa, m3, x2)
x4 = forward(Kappa, m4, x3)

G = [x0 x1 x2 x3 x4]
heatmap(transpose(G))
