import Random: shuffle, seed!
using Distributions
using Plots


# to have a reproducible example
seed!(0)

# sample 40 data points from multivariate normal distributions with a visible separation between them
points1 = rand(MvNormal([5, 8], 3 .* [1 3/5; 3/5 2]), 20)
points2 = rand(MvNormal([8, 5], 3 .* [1 3/5; 3/5 2]), 20)

# visualize the dataset
plt = scatter(points1[1, :], points1[2, :], label="y = 1")
scatter!(plt, points2[1, :], points2[2, :], label="y = -1")
display(plt)
readline()

# construct the dataset by labeling the data points and shuffling both classes
D = shuffle([
  tuple.(eachcol(points1), 1)
  tuple.(eachcol(points2), -1)
])
display(D)
readline()

# initialize trainable parameters: the normal vector and offset
w = [0; 1]
b = 0

# dot product can be expressed as w⋅x = w^T * x
hyperplane(x) = w' * x - b

# for drawing current hyperplane
function draw()
  plt = scatter(points1[1, :], points1[2, :], label="y = 1")
  scatter!(plt, points2[1, :], points2[2, :], label="y = -1")

  min_x = minimum(map((p) -> p[1][1], D))
  min_y = minimum(map((p) -> p[1][2], D))
  max_x = maximum(map((p) -> p[1][1], D))
  max_y = maximum(map((p) -> p[1][2], D))
  contour!(plt, min_x:0.1:max_x, min_y:0.1:max_y, (x, y) -> hyperplane([x, y]), levels=[-1], linestyles=:dash, colorbar_entry=false, color=:red)
  contour!(plt, min_x:0.1:max_x, min_y:0.1:max_y, (x, y) -> hyperplane([x, y]), levels=[0], linestyles=:solid, label="SVM prediction", colorbar_entry=false, color=:green)
  contour!(plt, min_x:0.1:max_x, min_y:0.1:max_y, (x, y) -> hyperplane([x, y]), levels=[1], linestyles=:dash, colorbar_entry=false, color=:blue)

  display(plt)
end

# show initial hyperplane
draw()
readline()

# loss function and the regularization parameter
λ = 3 / 100
loss() = λ * w' * w + sum([max(0, 1 - y * hyperplane(x)) for (x, y) in D]) / length(D)

# train the model on the dataset. α represents the learning rate
function fit(α=0.003)
  for (x, y) in D
    is_correct = y * hyperplane(x) >= 1

    # different gradients have to be applied based on the condition
    if is_correct
      # adjust according to the gradients scaled by the learning rate
      # gradients point to the steepest ascent, but we want to minimize the loss function, so we subtract the gradient
      global w -= α * 2λ * w
      global b -= α * 0
    else
      global w -= α * (2λ * w .- y * x)
      global b -= α * y
    end
  end
end

# run training once and show new result
fit()
draw()
println("Loss=$(loss())")
readline()


# train a few more times
for _ in 1:10
  fit()
end
draw()
println("Loss=$(loss())")
readline()

# train the model more times to show the margin being changed
for _ in 1:10000
  fit()
end
draw()
println("Loss=$(loss())")
readline()


# finally, we can perform a guess for a new point
# predictor function
p = sign ∘ hyperplane

x = [6, 7]
println("The point $x has a label $(p(x))")

readline()
