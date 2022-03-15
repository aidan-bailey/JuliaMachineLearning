module FFN

include("../Utils.jl")
include("../Data.jl")

using CUDA
using Flux
using Flux:  binarycrossentropy, mse

use_gpu = CUDA.functional()

# Hyperparameters:
learningRate = 7
epochs = 100
batchSize = 4
optimiser = Descent
loss = mse

@info "GPU: $use_gpu"
@info "Learning rate: $learningRate"
@info "Batch size: $batchSize"
@info "Epochs: $epochs"
@info "Optimiser: $optimiser"
@info "Loss: $loss"

model = Chain(
        Dense(2, 2, sigmoid),
        Dense(2, 1, sigmoid),
    )
if use_gpu
        model = fmap(cu, model)
end

train_dataloader, test_dataloader = Data.xor_dataloader(use_gpu, batchSize)

@info "TRAINING"
avgLoss = Utils.trainModel!(model, train_dataloader, learningRate, epochs, loss, optimiser)
@info "Avg. Loss: $avgLoss"
@info "TESTING"
avgLoss = Utils.testModel(model, test_dataloader, loss)
@info "Avg. Loss: $avgLoss"

end
