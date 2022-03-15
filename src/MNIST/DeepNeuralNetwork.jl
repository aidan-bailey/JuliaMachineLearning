module DeepNeuralNetwork

include("../Utils.jl")
include("../Data.jl")

using CUDA
using Flux
using Flux: crossentropy, argmax, mse, logitcrossentropy, Descent

use_gpu = CUDA.functional()

# Hyperparameters:
learningRate = 0.001
epochs = 200
batchSize = 64
optimiser = Descent
loss = mse

@info "GPU: $use_gpu"
@info "Learning rate: $learningRate"
@info "Batch size: $batchSize"
@info "Epochs: $epochs"
@info "Optimiser: $optimiser"
@info "Loss: $loss"

model = Chain(
    Dense(28 * 28, 32, tanh),
    Dense(32, 32, tanh),
    Dense(32, 10),
    softmax
)
if use_gpu
    model = fmap(cu, model)
end

train_dataloader, test_dataloader = Data.mnist_dataloaders(use_gpu, batchSize)

@info "TRAINING"
@time avgLoss = Utils.trainModel!(model, train_dataloader, learningRate, epochs, loss, optimiser)
@info "Avg. Loss: $avgLoss"
@info "TESTING"
@time avgLoss = Utils.testModel(model, test_dataloader, loss)
@info "Avg. Loss: $avgLoss"

end
