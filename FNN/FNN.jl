module FNN

mutable struct Perceptron
    ζ::Float64 # Weighted Sum
    γ::Float64 # Output
    Perceptron() = new(0, 0)
end

struct Layer
    x::Array{Perceptron,1} # Perceptron Nodes
    β::Float64 # Bias
    Layer(n::Int64) = new(fill(Perceptron(), n))
    Layer(n::Int64, β::Float64) = new(fill(Perceptron(), n), β)
end

function Base.length(layer::Layer)
    return length(layer.x)
end

function weightMatrix(layerSizes::Array{Int64, 1})
    if isempty(layerSizes) || length(layerSizes) == 1
        return Matrix{Float64}(undef, 0, 0)
    end
    frontSize = popfirst!(layerSizes)
    weightLayer = zeros(frontSize*first(layerSizes))
    weightMat = reshape(weightLayer, (first(layerSizes), frontSize))
    return push!([], weightMatrix(layerSizes))
end

println(weightMatrix([1,1,1]))

mutable struct FeedForwardNetwork
    layers::Array{Layer,1} # Perceptron Layers
    weights::Array{Matrix{Float64}, 1} # Weights
    FeedForwardNetwork(layers::Array{Int64, 1}) =
        new(
            map(Layer, layers),
            weightMatrix(layers)
        )
    FeedForwardNetwork(layerSizes::Int64...) = FeedForwardNetwork([layerSizes...])
end

#ffn = FeedForwardNetwork(2,3)

#Base.print_matrix(Base.stdout,ffn.weights)

end # module
