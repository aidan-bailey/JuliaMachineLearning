using Flux
using Flux: update!

actual(x) = 4x + 2

x_train, x_test = hcat(0:5...), hcat(6:10...)

println(x_train)
println(x_test)

y_train, y_test = actual.(x_train), actual.(x_test)

model = Dense(1, 1)

#loss(x, y) = ((y - model(x))) / 2
loss(x, y) = Flux.Losses.mse(model(x), y)

println("Loss pre-training: ", loss(x_train, y_train))

opt = Descent(0.1)
data = [(x_train, y_train)]
parameters = Flux.params(model)

function customTrain!(loss, ps, data, optimiser)
    local trainingloss
    avgloss = 0
    for d in data
        gs = gradient(ps) do
            trainingloss = loss(d...)
            avgloss += trainingloss
            return trainingloss
        end
        update!(optimiser, ps, gs)
    end
end

for epoch in 1:200
    customTrain!(loss, parameters, data, opt)
end

println("Loss post-training: ", loss(x_train, y_train))
