include("FilteringGuiding.jl")
using Plots

function ψ(u::Float64; τ = 0.1)
    exp(-τ * u)
end

function Kappa(x::Array{Int64,1}, i::Int64) #θ = (δ,μ,ν,λ) argument
    δ = 0.001
    μ = 0.6
    ν = 0.1
    λ = 2.5

    if x[i] == 1
        neighbours = [j for j in [i-2, i-1, i+1, i+2] if j > 0 && j <= length(x)]
        N = count(map(n -> n == 2, x[neighbours]))   # neighbour == 2 (:= 'infected')

        K = [(1. - δ) * ψ(λ * N),    1. - (1. - δ) * ψ(λ * N),    0.       ]
    end
    if x[i] == 2
        K = [0.,                     ψ(μ),                        1. - ψ(μ)]
    end
    if x[i] == 3
        K = [1. - ψ(ν),              0.,                          ψ(ν)     ]
    end
    ProbabilityDistribution(K)
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

function forward(K::Function, ms::Array{Message{Float64},1}, x::Array{Int64,1}) #, wins::Array{Float64,1})
    N = length(x)
    wnew = Array{Float64,1}(undef, N)
    xnext = Array{Int64,1}(undef, N)
    for i = 1:N
        xnext[i], wnew[i] = forward(K(x, i), ms[i], x[i]) #, wins[i])
    end
    xnext, wnew
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
N = length(x0)
w0 = zeros(N)

x1, w1 = forward(Kappa, m1, x0) #, w0)
x2, w2 = forward(Kappa, m2, x1) #, w1)
x3, w3 = forward(Kappa, m3, x2) #, w2)
x4, w4 = forward(Kappa, m4, x3) #, w3)

G = [x0 x1 x2 x3 x4]
heatmap(transpose(G))

likelihood = reduce(+, [w1; w2; w3; w4; [log(h0[i].likelihood[x0[i]]) for i = 1:N]])
println(likelihood)
