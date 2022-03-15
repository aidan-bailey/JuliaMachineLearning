module MLP

function dot(A::Matrix, B::Matrix)::Matrix{Float64} # This can be recursive
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

function activation(x::Float64)::Float64
    return sigmoid(x) >= 0.5 ? 1 : 0
end

mutable struct Perceptron
    z::Float64 # Weighted Sum
    x::Int64 # Output
    Perceptron() = new(0.0, 0)
end

struct Layer
    nodes::Array{Perceptron,1} # Perceptron Nodes
    β::Float64 # Bias
    Layer(n::Int64, β::Float64) = new(fill(Perceptron(), n), β)
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

struct MultiLayerPerceptron
    layers::Array{Layer,1} # Perceptron Layers
    weights::Array{Matrix{Float64},1} # Weights
    MultiLayerPerceptron(layers::Array{Int64,1}, biases::Array{Float64,1}) =
        new(
            map(
                ((layerSize, bias),) ->
                    Layer(layerSize, bias),
                zip(layers, biases)
            ),
            weightMatrix(layers)
        )
end

function loadInput!(mlp::MultiLayerPerceptron, input::Array{Float64,1})::Nothing
    inputLayer = first(mlp.layers)
    inputLength = length(input)
    inputLayerLength = length(inputLayer)
    if inputLength != inputLayerLength
        error("input length, $inputLength, does not match that of the input layer, $inputLayerLength")
    end
    for index in eachindex(inputLayer.nodes)
        inputLayer.nodes[index].x = input[index]
    end
end

function forwardPass!(mlp::MultiLayerPerceptron)::Nothing
    for (index, inputLayer) in enumerate(view(mlp.layers, 1:length(mlp.layers)-1))
        outputLayer = mlp.layers[index+1]
        outputSize = length(outputLayer)
        w = mlp.weights[index]
        x = permutedims(hcat(collect(map(node -> fill(node.x, outputSize), inputLayer.nodes))...))
        z = w ⋅ x .+ outputLayer.β
        γ = map(activation, z)
        for index in eachindex(γ)
            outputLayer.nodes[index].z = z[index]
            outputLayer.nodes[index].x = γ[index]
        end
    end
end

function backwardPass!(mlp::MultiLayerPerceptron, truth::Array{Int64,1}, α::Float64)::Nothing

end

mlp = MultiLayerPerceptron([2, 3, 2], [1.5, 1.5, 1.5])
loadInput!(mlp, [0.0, 0.0])
forwardPass!(mlp)
println(mlp)

end
