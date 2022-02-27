module LR

function dot(A::Matrix{Float64}, B::Matrix{Float64})::Matrix{Float64} # This can be recursive
    nrowA, _ = size(A)
    _, ncolB = size(B)
    if nrowA != ncolB
        error("Row count of matrix A, $nrowA, is not equal to column count of matrix B, $ncolB")
    end
    result = Float64[]
    for (rowA, colB) in zip(1:nrowA, 1:ncolB)
        push!(result, A[rowA, :]'B[:, colB])
    end
    return reshape(result, (nrowA, 1))
end

⋅ = dot

function sigmoid(x::Float64)::Float64
    return 1 / (1 + ℯ^(-x))
end

S = sigmoid

function tanh(z::Float64)::Float64
    return (ℯ^z - ℯ^-z) / (ℯ^z + ℯ^-z)
end

function relu(z::Float64)::Float64
    return max(z, 0)
end

function softmax(z::Array{Float64,1})
    return z / sum(z)
end

function crossEntropyLoss(prediction::Array{Float64,1}, truth::Array{Int64,1})
    loss = -sum([t * log(ℯ, p) for p in prediction, t in truth])
    grad = prediction - truth
    return loss, grad
end

Lce = crossEntropyLoss

mutable struct Perceptron
    z::Float64 # Weighted Sum
    x::Float64 # Output
    Perceptron() = new(0, 0)
end

struct Layer
    nodes::Array{Perceptron,1} # Perceptron Nodes
    β::Float64 # Bias
    activationFunction::Function # Activation Function
    lossFunction::Function
    Layer(n::Int64, β::Float64, activationFunction::Function, lossFunction::Function) =
        new(fill(Perceptron(), n), β, activationFunction, lossFunction)
end

function Base.length(layer::Layer)::Int64
    return length(layer.nodes)
end

function weightMatrix(layerSizes::Array{Int64,1})
    if isempty(layerSizes) || length(layerSizes) == 1
        return Array{Matrix{Float64}}[]
    end
    inputSize = popfirst!(layerSizes)
    outputSize = first(layerSizes)
    return append!(
        [reshape(zeros(inputSize * outputSize), (outputSize, inputSize))],
        weightMatrix(layerSizes)
    )
end

struct FeedForwardNetwork
    layers::Array{Layer,1} # Perceptron Layers
    weights::Array{Matrix{Float64},1} # Weights
    FeedForwardNetwork(layers::Array{Int64,1}, biases::Array{Float64,1}, activationFunctions, lossFunctions) =
        new(
            map(
                ((layerSize, bias, activationFunction, lossFunction),) ->
                    Layer(layerSize, bias, activationFunction, lossFunction),
                zip(layers, biases, activationFunctions, lossFunctions)
            ),
            weightMatrix(layers)
        )
end

function loadInput(ffn::FeedForwardNetwork, input::Array{Float64,1})::Nothing
    inputLayer = first(ffn.layers)
    inputLength = length(input)
    inputLayerLength = length(inputLayer)
    if inputLength != inputLayerLength
        error("input length, $inputLength, does not match that of the input layer, $inputLayerLength")
    end
    for index in eachindex(inputLayer.nodes)
        inputLayer.nodes[index].x = input[index]
    end
end

function forwardPass(ffn::FeedForwardNetwork, input::Array{Float64,1})::Nothing
    loadInput(ffn, input)
    for (index, inputLayer) in enumerate(view(ffn.layers, 1:length(ffn.layers)-1))
        outputLayer = ffn.layers[index+1]
        outputSize = length(outputLayer)
        w = ffn.weights[index]
        x = permutedims(hcat(collect(map(node -> fill(node.x, outputSize), inputLayer.nodes))...))
        z = w ⋅ x .+ outputLayer.β
        γ = map(outputLayer.activationFunction, z)
        for index in eachindex(γ)
            outputLayer.nodes[index].z = z[index]
            outputLayer.nodes[index].x = γ[index]
        end
    end
    outputNodes = last(ffn.layers).nodes
    γ = softmax(map(p -> p.x, outputNodes))
    for index in eachindex(γ)
        outputNodes[index].x = γ[index]
    end
end


function backpropagate(ffn::FeedForwardNetwork, truth::Array{Int64,1})
    for i in 1:length(ffn.layers)-1;
        outputLayer = ffn.layers[end+1-i]
        outputValues =
        inputLayer = ffn.layers[end-i]
        loss, grad = outputLayer.lossFunction(map(n -> n.x, last(ffn.layers).nodes), truth)
    end
end

function train(ffn::FeedForwardNetwork, input::Array{Float64,1}, truth::Array{Int64,1})#, η::Float64)
    forwardPass(ffn, input)
    backpropagate(ffn, truth)
end

train_set = [
    ([0.0, 0.0], [1, 0]),
    ([1.0, 0.0], [0, 1]),
    ([0.0, 1.0], [0, 1]),
    ([1.0, 1.0], [0, 0]),
]

ffn = FeedForwardNetwork([2, 3, 2], [1.5, 1.5, 1.5], [S, S, S], [Lce, Lce, Lce])

for (input, output) in train_set
    train(ffn, input, output)
end

end # module
