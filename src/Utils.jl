module Utils

using ProgressMeter
using Flux
using Flux: update!, binarycrossentropy, Optimiser, flatten, DataLoader, reset!

function customTrain!(loss, model, data, optimiser)
    local trainingloss
    parameters = params(model)
    for (x_batch, y_batch) in data
        gradients = gradient(parameters) do
            trainingloss = loss(model(x_batch), y_batch)
            return trainingloss
        end
        update!(optimiser, parameters, gradients)
    end
    return trainingloss
end

function trainModel!(model::Chain, dataloader::DataLoader, α::Number, epochs::Int64, loss, optimiser)::Float64

    testmode!(model, false)
    reset!(model)

    if epochs < 1
        error("Number of epochs, $epochs, is less than 1.")
    end

    @info "Training (α=$α, epochs=$epochs, loss=$loss, optimiser=$optimiser)"
    opt = optimiser(α)

    p = Progress(epochs; dt=1, desc="Training:", barglyphs=BarGlyphs("[=> ]"), showspeed=true)
    local trainingloss
    for epoch = 1:epochs
        #@info "Epoch: $(epoch)"
        trainingloss = customTrain!(loss, model, dataloader, opt)
        ProgressMeter.next!(p; showvalues = [(:Epoch,epoch), (:Loss, trainingloss)])
    end
    return trainingloss

end

function testModel(model::Chain, dataloader::DataLoader, loss)::Float64
    @info "Testing"
    testmode!(model, true)
    reset!(model)
    avgloss = 0
    for (x, y) in dataloader
        ŷ = model(x)
        testloss = loss(ŷ, y)
        avgloss += testloss
    end
    return avgloss/length(dataloader)
end

end
