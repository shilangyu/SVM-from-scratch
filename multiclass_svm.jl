using MLDatasets: MNIST

include("SVM.jl")
using .Models

preprocess(x, y) = (reshape(x, 28^2, size(x, 3)), y)
to_class(y, class) = (y == class) * 2 .- 1

train_x, train_y = preprocess(MNIST.traindata(Float32)...)
test_x, test_y = preprocess(MNIST.testdata(Float32)...)

svms = [
  LinearSVM(train_x, to_class.(train_y, i)) for i in 0:9
]


for svm in svms
  for _ in 1:100
    fit!(svm)
  end
end

function acc(svm, class)
  corr = 0
  for (x, y) in zip(eachcol(test_x), to_class.(test_y, class))
    if predict(svm, x) == y
      corr += 1
    end
  end

  corr / length(test_y)
end


error("TODO")
