include("FilteringGuiding.jl")
import StatsBase.sample, StatsBase.Weights, Plots.heatmap

struct Population{T<:HTransform{Float64}}
    htransforms::Array{T} #is there any reason not to only allow F64 instead of AbstractFloat
end

struct Messages{T<:Message{Float64}}
    messages::Array{T} #naming?
end

struct States{T<:Probability{Float64}}
    states::Array{T} #naming?
end

function backward(K::Kernel, hs::Population) ## Kernel, population {F64}?
    N = length(hs.htransforms)               ## in general: when to specify {T}
    ms = Array{Message{Float64}}(undef, N)
    houts = Array{HTransform{Float64}}(undef, N)
    for i in 1:N
        hout = HTransform(K.transitionmatrix * hs.htransforms[i].likelihood)
        ms[i] = Message(hs.htransforms[i], hout) # add index method to population hs[i]
        houts[i] = hout
        #ms[i], houts[i] = backward(K, hs.htransforms[i]) # UndefVarError: K not defined ???
    end
    Messages(ms), Population(houts)
end

function forward(K::Function, ms::Messages, x::States)
    I = length(x.states)
    Ns = Array{Float64,1}(undef, I) # not sure if fast but definitely isnt pretty
    # Neighbours on boundarys, infected neighbours will have all their mass in p[2] 'I'
    Ns[1]   = (x.states[2].p[2] == 1) + (x.states[3].p[2] == 1) #casting Int64 into Float64 array?
    Ns[2]   = (x.states[1].p[2] == 1) + (x.states[3].p[2] == 1) + (x.states[4].p[2] == 1)
    Ns[I-1] = (x.states[I-3].p[2] == 1) + (x.states[I-2].p[2] == 1) + (x.states[I].p[2] == 1)
    Ns[I]   = (x.states[I-2].p[2] == 1) + (x.states[I-1].p[2] == 1)
    #= in general there is a weird thing going on where its faster to push
    distributions forward (e.g. μ1 = μ0 * K) than to keep the state as an integer
    and sample from the appropriate row of: (e.g. μ0 = [1 0 0], μ1 = μ*K
                                             is equivalent to μ0 = 1, μ1 = K[:,1])
    However, storing the state in a probability distribution as above causes the
    state dependent K(x) to be a bit 'wonky'.=#

    for i = 3:I-2 #inner nodes
        s = 0
        for j = i-2:i+2
            s += (x.states[j].p[2] == 1)   #turn off bounds check j // unroll inner loop?
        end
        Ns[i] = s
    end

    xns = States([Probability(zeros(1,3)) for i=1:I]) #preallocate
    # might as well iterate over Ns[i], ms[i] ... if i∈1:I is in zip lol
    # add iterators! for m ∈ ms instead of ms.messages
    for (N,m,μ,i) in zip(Ns, ms.messages, x.states, 1:I) #this fwd implementation is a bit hacky
        Kstar = changeofmeasure(K(N), m)        #and not very general due to K(N(x))
        μout = μ.p * Kstar.transitionmatrix     #keep reusing K allocation?
        x = sample(Weights(vec(μout)))          #changeofmeasure!(K!(kernel, N)) ?
        xns.states[i].p[x] = 1.0
    end
    xns
end

function ψ(u::Float64)
    τ = 0.1
    exp(-τ * u)
end

function mykappa(N::Float64) # x ↦ N
    δ = 0.001
    μ = 0.6
    ν = 0.1
    λ = 2.5 # differentiate fixed Λ from dependent λN(x)

    K = [(1-δ)*ψ(λ*N)   1-(1-δ)*ψ(λ*N)   0
         0              ψ(μ)             1-ψ(μ)
         1-ψ(ν)         0                ψ(ν)   ]
    Kernel(K)
end

δ = 0.001
μ = 0.6
ν = 0.1
Λ = 0.5 # differentiate fixed Λ from λN(x)

Ktilde = Kernel([(1-δ)*ψ(Λ)   1-(1-δ)*ψ(Λ)   0
                  0           ψ(μ)           1-ψ(μ)
                  1-ψ(ν)      0              ψ(ν)   ])

h4a = HTransform([0., 0., 1.]) # 'R'
h4b = HTransform([0., 0., 1.]) # 'R'
h4c = HTransform([0., 1., 0.]) # 'I'
h4d = HTransform([0., 0., 1.]) # 'R'
h4e = HTransform([0., 0., 1.]) # 'R'

h4 = Population([h4a, h4b, h4c, h4d, h4e]) # observations

m4, h3 = backward(Ktilde, h4)
m3, h2 = backward(Ktilde, h3)
m2, h1 = backward(Ktilde, h2)
m1, h0 = backward(Ktilde, h1)

μ0a = Probability([0. 1. 0.]) # 'I'
μ0b = Probability([1. 0. 0.]) # 'S'
μ0c = Probability([1. 0. 0.]) # 'S'
μ0d = Probability([1. 0. 0.]) # 'S'
μ0e = Probability([1. 0. 0.]) # 'S'

μ0 = States([μ0a, μ0b, μ0c, μ0d, μ0e]) # starting 'distributions'

μ1 = forward(mykappa, m1, μ0)
μ2 = forward(mykappa, m2, μ1)
μ3 = forward(mykappa, m3, μ2)
μ4 = forward(mykappa, m4, μ3)

G = zeros(5,5)
for (μ, i) in zip([μ0, μ1, μ2, μ3, μ4], 1:5)
    G[i,:] = [findall(x->x==1.0, vec(individual.p))[1] for individual in μ.states]
end
heatmap(G)
#print(G)  just print G from console (it displays correct dimension structure)
#need to find/make a pcolormesh equivalent Plots.plot(Plots.heatmap(G))
