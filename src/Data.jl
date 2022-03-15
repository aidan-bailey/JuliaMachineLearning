module Data

using ProgressMeter
using CUDA
using GZip
using Flux: fmap, DataLoader, flatten, onehotbatch

function xor_dataloaders(cuda::Bool, batchSize::Int64, shuffle::Bool = true, partial::Bool = false)
    x_set = [0 0 1 1; 0 1 0 1]
    y_set = [0;; 1;; 1;; 0]
    if cuda
        x_set = fmap(cu, x_set)
        y_set = fmap(cu, y_set)
    end
    return DataLoader(
        (data = x_set, label = y_set),
        batchsize = batchSize,
        shuffle = shuffle,
        partial = partial
    ), DataLoader(
        (data = x_set, label = y_set),
        batchsize = batchSize,
        shuffle = false,
        partial = false
    )
end

function mnist_dataloaders(cuda::Bool, batchSize::Int64, shuffle::Bool = true, partial::Bool = false)
    function parseidx(data)
        type_constructors = [UInt8,
            Int8,
            Int16,
            Int16,
            Int32,
            Float32,
            Float64]
        idxtype = type_constructors[data[3]-0x07]
        dimensions = data[4]
        sizes = map(i -> reinterpret(UInt32, reverse(data[(4+(i-1)*4+1):(4+i*4)]))[1], 1:dimensions)
        reshape(convert(Array{idxtype}, data[4*(dimensions+1)+1:end]), Tuple(reverse(sizes)))
    end

    local x_train, y_train, x_test, y_test

    datasetpath = joinpath("Datasets", "MNIST")

    if !isdir(datasetpath)
        mkpath(datasetpath)
    end

    processedfiles = readdir(datasetpath)

    url = "http://yann.lecun.com/exdb/mnist"
    names = [("train-images-idx3-ubyte.gz", "train-images.idx3-ubyte"),
             ("train-labels-idx1-ubyte.gz", "train-labels.idx1-ubyte"),
             ("t10k-images-idx3-ubyte.gz", "t10k-images.idx3-ubyte"),
             ("t10k-labels-idx1-ubyte.gz", "t10k-labels.idx1-ubyte")]

    for (zipname, unzipname) in names
        if unzipname in processedfiles
            continue
        end
        @debug "Fetching $unzipname..."
        zipfilepath = joinpath(datasetpath, zipname)
        download("$url/$zipname", zipfilepath)
        unzipfilepath = joinpath(datasetpath, unzipname)
        filestream = GZip.open(zipfilepath)
        write(unzipfilepath, read(filestream))
        close(filestream)
        rm(zipfilepath)
    end

    open("Datasets/MNIST/train-images.idx3-ubyte", "r") do f
        data = parseidx(read(f))
        data = flatten(data)
        x_train = reshape(data, (28 * 28, 60000))
        x_train = convert(Matrix{Float32}, x_train)
        if cuda
            x_train = fmap(cu, x_train)
        end
    end

    open("Datasets/MNIST/train-labels.idx1-ubyte", "r") do f
        data = parseidx(read(f))
        y_train = onehotbatch(data, 0:9)
        if cuda
            y_train = fmap(cu, y_train)
        end
    end

    open("Datasets/MNIST/t10k-images.idx3-ubyte") do f
        data = parseidx(read(f))
        data = flatten(data)
        x_test = reshape(data, (28 * 28, 10000))
        x_test = convert(Matrix{Float32}, x_test)
        if cuda
            x_test = fmap(cu, x_test)
        end
    end

    open("Datasets/MNIST/t10k-labels.idx1-ubyte") do f
        data = parseidx(read(f))
        y_test = onehotbatch(data, 0:9)
        if cuda
            y_test = fmap(cu, y_test)
        end
    end

    train_dataloader = DataLoader(
        (data = x_train, label = y_train),
        batchsize = batchSize,
        shuffle = shuffle,
        partial = partial
    )
    test_dataloader = DataLoader(
        (data = x_test, label = y_test),
        batchsize = 1,
        shuffle = false,
        partial = false
    )

    return train_dataloader, test_dataloader

end

end
