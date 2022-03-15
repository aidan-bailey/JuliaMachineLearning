using Flux
using Flux: train!

actual(x) = 4x + 2

x_train, x_test = hcat(0:5...), hcat(6:10...)

println(x_train, x_test)

y_train, y_test = actual.(x_train), actual.(x_test)

println(y_train, y_test)

model = Dense(1, 1)

println("Weight: $(model.weight)")

println("Bias: $(model.bias)")

println("Predictions: $(model(x_train))")

loss(x, y) = Flux.Losses.mse(model(x), y)

println("Loss pre-training: ", loss(x_train, y_train))

opt = Descent(0.1)
data = [(x_train, y_train)]

parameters = Flux.params(model)

for epoch in 1:200
    println("Epoch $epoch")
    train!(loss, parameters, data, opt)
end

println("Loss post-training: ", loss(x_train, y_train))
