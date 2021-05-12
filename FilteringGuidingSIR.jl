using StaticArrays, LinearAlgebra, BenchmarkTools, Plots, Distributions #, Random, StatsBase

struct TransitionKernel{N, T, L}
    transitionmatrix::SMatrix{N, N, T, L}
    grad::SMatrix{N, N, T, L}
end

struct HTransform{N, T}
    likelihood::MVector{N, T}
    grad::MVector{N, T}
    #HTransform() = new(@MVector zeros(Float64, 3), @MVector zeros(Float64, 3))
end

function HTransform(state::Int64, N, T)
    h = @MVector zeros(T, N)
    h[state] = 1.0
    HTransform(h, @MVector zeros(T, N))
end

struct Message{N, T}
    h::HTransform{N, T}
    pullback::HTransform{N, T}
end

# pointwise operations slow on SArrays, remove .* operations ?
Base.:*(h1::HTransform, h2::HTransform) = HTransform(h1.likelihood .* h2.likelihood,
                                          h1.likelihood .* h2.grad + h1.grad .* h2.likelihood)

Base.:*(K::TransitionKernel, h::HTransform) = HTransform(K.transitionmatrix * h.likelihood,
                                              K.transitionmatrix * h.grad + K.grad * h.likelihood)

function backward(K::TransitionKernel, h::HTransform) # is this type stable?
   pullback = K * h    # does Julia infer the static type of the pullback, messages etc?
   Message(h, pullback), pullback
end

function backward!(K::TransitionKernel, ms::Vector{Message{3, Float64}}, hs::Vector{HTransform{3, Float64}})
    for i in eachindex(ms)
        ms[i], hs[i] = backward(K, hs[i])
    end
    nothing
end

function backwardfiltering(Kappa::TransitionKernel, Xend::Array{Int64}, Xobs::Matrix{Int64}, dims::Tuple{Integer, Integer})
    N, T = dims                               # this is probably a dumb way of allocating
    ms = [[zerosMsg for i = 1:N] for j = 1:T] # save for next MCMC iterations
    h = [HTransform(x, 3, Float64) for x ∈ Xend]

    h̃, dh̃ = 0.0, 0.0
    for i = T:-1:1
        backward!(Kappa, ms[i], h) # slower than for i in eachindex(ms) backward!(K, m[i], h[i]) end ?

        if i % 50 == 1 #generalize this: if i ∈ obs
            X = Xobs[:,(i÷50)+1]
            h̃obs, dh̃obs = ℓ(h, X)
            h̃ += h̃obs
            dh̃ += dh̃obs
            hobs = [HTransform(x, 3, Float64) for x in X]
            h = h .* hobs #write a faster 'observation' function? 'one hot'
        end
    end
    ms, h̃, dh̃
end

function ℓ(h::Array{HTransform{3, Float64}}, x::Vector{Int64})
    ls  = [h[i].likelihood[x[i]] for i in eachindex(x)] # dont need to allocate
    gradls = [h[i].grad[x[i]] for i in eachindex(x)]    # keep running count w/ generator / 'channel'
    logl  = sum(log.(ls))
    gradlogl = sum(gradls ./ ls)       # dlog(h[x])/dθ = 1 / h[x] * dh[x]/dθ  ?
    logl, gradlogl
end
#=
function fusion(branches::Vector{HTransform})
    reduce(*, branches)
end

function fusion(branches::Array{Array{HTransform, 1}, 1})
    [fusion(collect(hs)) for hs in zip(branches...)]
end
=#
function forward(K::TransitionKernel, m::Message, x::Integer, z::Float64)
    p = K.transitionmatrix[x,:]        # correct row, slow!
    v = p .* m.h.likelihood            # change of measure, elementwise slow!
    wnew = dot(p, m.h.likelihood) / m.pullback.likelihood[x]

    dp = K.grad[x,:]

    dwnew = ( ( dot(dp, m.h.likelihood) + dot(p, m.h.grad) ) * m.pullback.likelihood[x] -
                dot(p, m.h.likelihood) * m.pullback.grad[x] ) / (m.pullback.likelihood[x]^2)

    zscaled = z * sum(v)
    if zscaled == 0. println("CONDITIONED THROUGH ZERO-LIKELIHOOD STATE") end

    sample = zscaled < v[1] ? 1 : (zscaled < v[1] + v[2] ? 2 : 3)

    sample, log(wnew), dwnew/wnew     # dlog(w(x°))/dθ = 1/w(x°) * dw(x°)/dθ
end

function forwardguiding(Kappa::Vector{TransitionKernel{3, Float64, 9}}, x0::Vector{Int64}, ms::Vector{Vector{Message{3, Float64}}}, Z::Matrix{Float64}, dims::Tuple{Integer, Integer})
    N, T = dims
    w, dw = 0.0, 0.0

    X = Array{Int64, 2}(undef, (N, T+1))
    X[:,1] = x0

    for i = 1:T
        x = X[:,i] # #map(i->infectedneighbours(x, i), i for i = 1:N) ?
        m = ms[i]
        for j = 1:N
            X[j,i+1], wnew, dwnew = forward(Kappa[statetoindex(x, j)], m[j], x[j], Z[j,i])
            w += wnew
            dw += dwnew
        end
    end
    X, w, dw
end

function Kappa(θ::NTuple{4, Float64}, Ñ) # local estimate Ñ backward filtering
    δ, μ, ν, λ = θ; τ = 0.1       #first run is just Λ = 0.5 ⟹ Ñ = 0.2 ∀ (i, t)
    ψ(u) = exp(-τ*u)
    dψ(u) = -τ*ψ(u)

    K = SMatrix{3,3}([(1.0-δ)*ψ(λ*Ñ)   1.0-(1.0-δ)*ψ(λ*Ñ)   0.
                       0.              ψ(μ)                 1.0-ψ(μ)
                       1.0-ψ(ν)        0.                   ψ(ν)   ])

    ∂K∂ν = SMatrix{3,3}([0.     0.     0.
                         0.     0.     0.
                        -dψ(ν)  0.     dψ(ν)])

    TransitionKernel(K, ∂K∂ν)
end

function statetoindex(x::Array{Int64, 1}, i::Int64)
    I = length(x)
    if i == 1       neighbours = [2, 3]
    elseif i == 2   neighbours = [1, 3, 4]
    elseif i == I-1 neighbours = [I-3, I-2, I]
    elseif i == I   neighbours = [I-2, I-1]
    else            neighbours = [i-2, i-1, i+1, i+2]
    end
    N = sum(x[neighbours] .== 2)
    return N+1
end

function generate(Kappa::Vector{TransitionKernel{3, Float64, 9}}, x0::Array{Int64, 1}, dims::Tuple{Integer, Integer})
    N, T = dims
    Z = rand(Float64, dims)
    X = Array{Int64, 2}(undef, (N, T+1))
    X[:,1] = x0
    for j = 1:T
        x = X[:,j]
        for i = 1:N
            p = Kappa[statetoindex(x, i)].transitionmatrix[x[i],:]
            u = Z[i,j]
            X[i,j+1] = u < p[1] ? 1 : (u < p[1] + p[2] ? 2 : 3)
        end
    end
    X, Z
end


# these are some dumb constants because i can't get constructors to work
const zerosMVec = @MVector zeros(3)
const zerosHTf  = HTransform(zerosMVec, zerosMVec)
const zerosMsg  = Message(zerosHTf, zerosHTf)

function main()
    # generate data, even left most individuals infected
    x0 = [[2 for i = 1:7]; [1 for i = 1:93]]
    dims = (length(x0), 500)

    # all possible neighborhood-dependent versions of Kappa
    θtrue = (0.001, 0.6, 0.1, 2.5)
    Ktrue = [Kappa(θtrue, i) for i = 0:4]

    Xtrue, Ztrue = generate(Ktrue, x0, dims)

    # find a general way to do this
    #Xobs = Dict((i, j) => X[i,j] for i=1:100, j=1:50:451)
    Xend = Xtrue[:,end]
    Xobs = Xtrue[:, [i for i in 1:50:501]]

    # do some inference
    θs(ν) = (0.001, 0.6, ν, 2.5)

    ν = 0.15

    θ = θs(ν)
    K = [Kappa(θ, i) for i = 0:4] # all possible N[x] states
    Z = rand(Float64, dims)

    ms, h̃ = backwardfiltering(Kappa(θ, 0.2), Xend, Xobs, dims)
    X, w = forwardguiding(K, x0, ms, Z, dims); X0 = X
    h = h̃ + w

    ρ = 0.995 # innovation exploration rate
    propσ = 0.2 # proposal variance

    νs = [ν]
    for i = 1:3000
        if mod(i, 100) == 0 println(i) end
        Z° = cdf.(Normal(), ρ*quantile.(Normal(), Z) + √(1 - ρ^2)*randn(Float64, dims))

        ν° = ν * exp(propσ*randn()) # proposal
        θ° = θs(ν°)

        K° = [Kappa(θ°, i) for i = 0:4]

        ms, h̃° = backwardfiltering(Kappa(θ°, 0.2), Xend, Xobs, dims)
        X°, w° = forwardguiding(K°, x0, ms, Z°, dims)
        h° = h̃° + w°

        if log(rand()) < (h° - h) # also include prior / proposal likelihood
            X, X° = X°, X
            Z, Z° = Z°, Z
            h, h° = h°, h
            ν = ν°
            println(h)
        end
        push!(νs, ν)
    end
    νs, X0, X, Xtrue
end

νs, X0, X, Xtrue = main()

histogram(νs, bins=20)     # still something wrong somewhere..
heatmap(transpose(X0))
heatmap(transpose(X))
heatmap(transpose(Xtrue))

#=
function HTransform(states::Array{Int64}, N::Integer)
    T = Float64
    Vector{HTransform{N, T}}([HTransform(state, N) for state ∈ states])
end

why does this take 100x as long ?

Vector{HTransform{3, Float64}}([HTransform(1, 3) for i = 1:10])  ?

julia> typeof(state)
Vector{Int64} (alias for Array{Int64, 1})

julia> @btime state[1]
  13.331 ns (0 allocations: 0 bytes)
1

julia> @btime HTransform(1, 3, Float64)
  11.705 ns (2 allocations: 64 bytes)
HTransform{3, Float64}([1.0, 0.0, 0.0], [0.0, 0.0, 0.0])

julia> @btime HTransform(state[1], 3, Float64)
  2.424 μs (21 allocations: 1.22 KiB)                       ???
HTransform{3, Float64}([1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
=#

#=
function ℓ(h::Array{HTransform}, x::Vector{Int64}) #why does this do MORE allocations
    h̃obs  = reduce(+, (log(h[i].likelihood[x[i]]) for i in eachindex(x)))
    dh̃obs = reduce(.+, (log.(h[i].grad[x[i]]) for i in eachindex(x)))
    h̃obs, dh̃obs/h̃obs
end
=#

#=
Need to write some sort of statedependency -> [Kappa(θ) for θ in Θ] container
need to do backward filtering with an arbitrary N[t,i] ↦ Kappa but also for precomputed struct

precalculate each individuals neighbours -> more complex neighborhood structure

anonymous function Kappa(θ...)

#treating x0 differently than other conditioned points seems arbitrary
=#
