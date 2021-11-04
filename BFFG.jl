using StaticArrays, LinearAlgebra

abstract type AbstractBayesNet end

struct Message{N, T1, T2} #should this be parametric of AbstractFloat type
    htransform::MVector{N, T1}
    pullback::Vector{Vector{T2}}
end

struct InteractingParticleSystem <: AbstractBayesNet
    """
    An interacting particle system is distringuisable from more general BNs
    because for each i we enforce nodes (i,t), t∈[1:T] to be of the 'same' process,
    meaning they have the same statespace, set of parents and cpds.

    Processes i may have different characteristics between them, however.

    An IPS is thus fully described by a mapping of each process i to these objects.

    In accordance with the paper Automatic Backward Filtering Forward Guiding ...
    we denote a fixed root node for each process.

    If the set of processes allows for an ordering such that if process i
    interacts with process j_1 and j_2, then it also interacts with all processes
    j ∈ [j_1, j_2]. The interacting process ids can then be described by a UnitRange
    -> fast indexing from samples Matrix

    Alternatively, use Dicts describing process dynamics (statespace, parents, cpds, root)
    to construct DiscreteNode <: AbstractNode structs. This is more formal but slower.
    There is no need for an actual Node struct, hashmaps work fine.
    """
    statespace::Dict{Int, Vector{Enum}}
    parents::Dict{Int, UnitRange{Int}}
    cpd::Dict{Int, Dict{Vector{Int}, Vector{AbstractFloat}}}
    cpdmatrix::Dict{Int, AbstractMatrix}
    root::Vector{Int}
    N::Int
    T::Int
end

function InteractingParticleSystem(statespace, parents, (cpd, cpdmatrix), root::Vector, (N, T))
    InteractingParticleSystem(statespace, parents, cpd, cpdmatrix, Int.(root), N, T)
end

function samplefrom(IPS::InteractingParticleSystem, Z::Matrix)
    """
    Samples from an IPS as an image of a vector of U(0, 1) random numbers,
    useful for reproduction and normalising flow purposes.

    Samples initialised with the root states.

    Formally we should cast the sample to the appropriate State, but in the
    SIR example all processes have the same statespace so we can skip this.

    e.g.: samples[i,t] = statespace[i][discretesample(p, Z[i,t])]

    Sampling could be done faster as condtional cumulative pds but doesnt allow c.o.m.
    """
    N, T = IPS.N, IPS.T
    samples = Matrix{Int}(undef, N, T)
    samples[:,1] = IPS.root
    for t in 2:T
        for i in 1:N
            p = IPS.cpd[i][samples[IPS.parents[i],t-1]]
            samples[i,t] = discretesample(p, Z[i,t])
        end
    end
    samples
end

function forwardguiding(IPS::InteractingParticleSystem, messages::Dict{Tuple{Int, Int}, Message}, Z::Matrix)
    """
    Similar to samplefrom() but performs a change of measure
    """
    N, T = IPS.N, IPS.T
    samples = Matrix{Int}(undef, N, T)
    samples[:,1] = IPS.root
    for t in 2:T
        for i in 1:N
            p = IPS.cpd[i][samples[IPS.parents[i],t-1]]
            p° = p .* messages[(i,t)].htransform
            samples[i,t] = discretesample(p°, sum(p°)*Z[i,t])
        end
    end
    samples
end

#=
function min(p, i, s, z)
    s+= p[i]
    s > z ? i : min(p, i+1, s, z)
end
=#
function discretesample(p::Vector, z)
    """
    Generalise this to length(p) without cumulative sum, e.g.
    min(p, 1, 0, z) using function above
    """
    z < p[1] ? 1 : z < p[1] + p[2] ? 2 : 3
end

function backwardfiltering(IPS::InteractingParticleSystem, parentsdensity, observations)
    """
    In the SIR case each process has an individual emission process with known
    parameters, so exact filtering is possible (no bw-marginalisation is required)
    and weights are equal to 1. In more general models observation 'nodes'
    should also have a message, because a weight must be computed.

    Current scheme: IPS is described by cpds per layer, and observation 'nodes'
    are treated seperately because they dont occur in each time slice. More generally
    observation 'nodes' should be treated the same way as latent nodes.

    H TRANSFORMS CREATED WHEN NEEDED AND DELETED LATER, SLOW?

    Formally bw-marginalisation scheme should be derived from parents' statespace,
    however, they are all equal in SIR so we do this more efficiently.

    pullbacks = backwardmarginalisation(cpullback, IPS.parentsdensity[(i,t)], [IPS.statespace[j] for j in IPS.parents[i]])
    """
    N, T = IPS.N, IPS.T
    #htransforms = Dict((i,t) => MVector{3, AbstractFloat}(ones(3)) for i=1:N, t=1:T)

    #this is not general, MVector will be of different size (not 3) in general DAGs.
    htransforms = Dict{Tuple{Int, Int}, MVector}()
    messages = Dict{Tuple{Int, Int}, Message}()

    # Should include obsparentsdensity if obs pullback should be bw-marginalised
    obsparents, obscpds, obsstate = observations

    for (id, obs) in pairs(obsstate)
        chtransform = obscpds[id][:,obs]
        #pullbacks = backwardmarginalisation(chtransform, obsparentsdensity[id])
        #for (parent, pullback) in zip(obsparents[id], pullbacks)
        #    htransforms[parent] *= pullback
        #end
        get!(htransforms, obsparents[id], chtransform)
    end

    for t in T:-1:2
        for i in N:-1:1
            htransform = htransforms[(i,t)]
            # this scaling is wrong, should be s.t. tensor prod of vectors ~ original. e.g. take roots of scaling factor
            # htransform = htransform ./ (sum(htransform))
            cpullback = IPS.cpdmatrix[i]*htransform
            pullbacks = backwardmarginalisation(cpullback, parentsdensity[(i,t)])
            for (parent, pullback) in zip(IPS.parents[i], pullbacks)
                htarget = get!(htransforms, (parent,t-1), MVector{3, AbstractFloat}(ones(3)))
                htransforms[(parent,t-1)] = MVector{3}(htarget .* pullback)
            end
            messages[(i,t)] = Message(htransform, pullbacks)
            delete!(htransforms, (i,t))
        end
    end
    messages
end

function mask(c, C)
    """
    I can explain this with some pen and paper
    """
    N = 3^C
    M = zeros((3, N))
    Δ = 3^(c-1)                       # length of a 'line'
    for i = 1:3
        for j = 1:3^(C-c)                 # how many 'lines' per row. e.g. 1:9 in 1:27 = 1 line
            M[i, 1 + (j-1)*(3^c) + (i-1)*Δ : 1 + (j-1)*(3^c) + i*Δ - 1] .= 1
        end
    end
    M
end

function backwardmarginalisation(cpullback, parentsdensity)
    """
    This isn't general yet, but works for N parents if they have state space size 3
    """
    E = 3
    C = Integer(round(log(length(parentsdensity)) / log(E)))

    jhtransform = cpullback .* parentsdensity

    s = sum(jhtransform) #test s and c
    c = s^(1/C)

    marginals = map(i -> mask(i, C) * parentsdensity, 1:C)
    unnormalisedprojections = map(i -> mask(i, C) * jhtransform ./ s, 1:C)

    [c .* projection ./ marginal for (projection, marginal) in zip(unnormalisedprojections, marginals)]
end

function buildprior(IPS::InteractingParticleSystem, messages, iterations)
    """
    Similar to forwardguiding() but keeps track of sample correlations
    """
    N, T = IPS.N, IPS.T
    samples = Matrix{Int}(undef, N, T)

    #myzeros(i) = i in [1, N] ? zeros(27) : i in [2, N-1] ? zeros(81) : zeros(243)
    parentsdensity = Dict((i,t) => 0.1 * normalisedones(i) for i=1:N, t=1:T)

    for k=1:iterations
        Z = rand(N, T)

        samples[:,1] = IPS.root
        for t in 2:T
            for i in 1:N
                parentsamples = samples[IPS.parents[i],t-1]
                parentsdensity[(i,t)][dot(parentsamples.-1, [3^l for l=0:length(parentsamples)-1])+1] += 0.9/iterations

                p = IPS.cpd[i][parentsamples]
                p° = p .* messages[(i,t)].htransform
                samples[i,t] = discretesample(p°, sum(p°)*Z[i,t])
            end
        end
    end
    parentsdensity
end

### USER FILE
"""
An IPS fully described by its root state and each process's conditional transition probabilities.
Because a process's parent set and cpds are constant at each time step, the object
is fully defined by a mapping from each process i to its statespace, parents and a cpd.

In the SIR case there are formally five seperate cpds. The 'left-most' process is missing
its two neighbours to the left, the process to the right is missing only its left
neighbours left neighbour, and so on. All interior nodes have the same cpd.

By defining all relevant pdfs and mapping each interior node to the same interior cpd we
decrease memory cost. All cpds are stored as hash maps, instead of computing probability
mass functions as they are required. This speeds up sampling significantly.
Alternatively @memoize the cpd function output. (This was quite slow when I tested it!)
"""
@enum State::UInt8 _S_=1 _I_=2 _R_=3

# Problem dimensions. Other CHMMs may feature interacting processes with differing statespace
N = 10
T = 101
dims = (N, T)

# All processes have the same statespace in this example problem.
E = [_S_, _I_, _R_]

# The 'base' transition kernel. @match not neccesary here but might by useful for other cpds
using Match
function K(i, parentstates::Tuple{Vararg{State}}, θ)
    δ, μ, ν, λ = θ
    τ = 0.1

    neighbours = setdiff(1:length(parentstates), i)
    N_i = sum(parentstates[neighbours] .== _I_)

    ψ(u) = exp(-τ*u)

    @match Int(parentstates[i]) begin
        1 => [(1.0-δ)*ψ(λ*N_i),   1.0-(1.0-δ)*ψ(λ*N_i),   0.          ]
        2 => [ 0.,                ψ(μ),                   1.0-ψ(μ)    ]
        3 => [ 1.0-ψ(ν),          0.,                     ψ(ν)        ]
    end
end

# Map from process id to its cpd transition matrix and pre-computed hash map, differentiable w.r.t θ
function cpds(θ, N)
    cpd(directparent, numberofparents, θ) = Dict{Vector{Int}, Vector{AbstractFloat}}([Int(s) for s in combination] => K(directparent, combination, θ) for combination in Iterators.product(ntuple(i -> E, numberofparents)...))
    cpd2K(cpd, C) = SMatrix{3^C, 3}(reduce(hcat, [cpd[[Int(pa) for pa in combination]] for combination in Iterators.product(ntuple(i -> E, C)...)])')
    cpd1 = cpd(1, 3, θ); K1 = cpd2K(cpd1, 3)   # i==1         # SMatrix{length(E)^C, length(E)}(cpd2K(cpd, C)) ....
    cpd2 = cpd(2, 4, θ); K2 = cpd2K(cpd2, 4)   # i==2
    cpdI = cpd(3, 5, θ); KI = cpd2K(cpdI, 5)   # interior
    cpdM = cpd(3, 4, θ); KM = cpd2K(cpdM, 4)   # i ==N-1
    cpdN = cpd(3, 3, θ); KN = cpd2K(cpdN, 3)   # i ==N
    # HOW TO MAKE HASHMAP POINT TO THE SAME OBJECTS WITHOUT COPYING?
    cpddict = merge!(Dict(1 => cpd1, 2 => cpd2, N-1 => cpdM, N => cpdN), Dict(i => cpdI for i=3:N-2))
    Kdict = merge!(Dict(1 => K1, 2 => K2, N-1 => KM, N => KN), Dict(i => KI for i=3:N-2))

    cpddict, Kdict
end

# Map from process id to statspace. In general each process could have a different E_i
statespace = Dict(i=> E for i=1:N)

# Map from process id to set of parents' process ids
parents = Dict(i => intersect(i-2:i+2, 1:N) for i=1:N)

# Fixed roots
root = vcat([_I_ for i=1:7], [_S_ for i=8:N])

# SIR model fully defined in terms of θ
SIR(θ) = InteractingParticleSystem(statespace, parents, cpds(θ, N), root, dims)

# Easy to do proposals e.g. SIR(θ')
θtrue = (0.001, 0.6, 0.1, 2.5)
G = SIR(θtrue)

# Conditional sample will be an image of (θ, Z), allows for normalising flow
S = samplefrom(G, rand(Float64, dims))

# Observation interval
τobs = 25

# Observations parents set. In this case each leaf has a direct parent. So trivial.
obsparents = Dict((i,t) => (i,t) for i=1:N, t=1:τobs:T)

# Observation emission process
Y = SMatrix{3, 3}(Matrix(1.0I, 3, 3))
obscpds = Dict((i,t) => Y for (i,t) in keys(obsparents))

# Observation states
obsstate = Dict((i,t) => S[i,t] for (i,t) in keys(obsparents))

# Observations tuple to be used as argument
obs = (obsparents, obscpds, obsstate)

# Prior parentsdensity. Processes on the edges have fewer parents -> smaller density
normalisedones(i) = i ∈ [1, N] ? ones(27)/27 : i ∈ [2, N-1] ? ones(81)/81 : ones(243)/243
parentsdensity = Dict((i,t) => normalisedones(i) for i=1:N, t=1:T)

# Messages assuming constant 'flat' prior
ms = backwardfiltering(G, parentsdensity, obs)

# Build better prior from samples and backwardfilter again. You can iterate these lines a few times.
parentsdensity = buildprior(G, ms, 5000)
ms = backwardfiltering(G, parentsdensity, obs)

# Sample guided process with 'better' messages
S2 = forwardguiding(G, ms, rand(Float64, dims))

# Plot: Black == Susceptible, Red == Infected, Yellow == Recovered
using Plots
heatmap(S)
heatmap(S2)

#using BenchmarkTools
#@btime samplefrom(G, rand(Float64, dims))

#@profiler backwardfiltering(G, parentsdensity, obs) combine = true
