import Random: shuffle, seed!
using Distributions
using Plots
import LinearAlgebra: ⋅

# ported from Matlab's implementation https://www.mathworks.com/matlabcentral/fileexchange/41459-6-functions-for-generating-artificial-datasets
function two_spirals(n_samples; noise::Float64=0.2)
  start_angle = π / 2
  total_angle = 3π

  N1 = floor(Int, n_samples / 2)
  N2 = n_samples - N1

  n = start_angle .+ sqrt.(rand(N1, 1)) .* total_angle
  d1 = [-cos.(n) .* n + rand(N1, 1) .* noise sin.(n) .* n + rand(N1, 1) .* noise]

  n = start_angle .+ sqrt.(rand(N2, 1)) .* total_angle
  d2 = [cos.(n) .* n + rand(N2, 1) * noise -sin.(n) .* n + rand(N2, 1) * noise]

  return d1, d2
end


# to have a reproducible example
seed!(0)

# sample 100 data points from multivariate normal distributions with a visible separation between them
points1 = [rand(MvNormal([10, 0], 1), 20) rand(MvNormal([0, 10], 1), 20)]
points2 = [rand(MvNormal([10, 10], 1), 20) rand(MvNormal([0, 0], 1), 20)]

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
# for more efficient access
X = [x for (x, y) in D]
Y = [y for (x, y) in D]
readline()

# create a kernel function with its hyperparameter
γ = 1 / 5
k(x, y) = exp(-γ * (x - y)' * (x - y))

# regularization parameter (essentially λ⁻¹)
C = 1000
# multipliers to be found using SMO
α = zeros(length(D))
b = 0

# dot product replaced with the kernel function
hyperplane(x) = (α .* Y) ⋅ k.(X, Ref(x)) + b

# for drawing current hyperplane
function draw()
  plt = scatter(points1[1, :], points1[2, :], label="y = 1")
  scatter!(plt, points2[1, :], points2[2, :], label="y = -1")

  min_x = minimum(map((p) -> p[1], X))
  min_y = minimum(map((p) -> p[2], X))
  max_x = maximum(map((p) -> p[1], X))
  max_y = maximum(map((p) -> p[2], X))
  contour!(plt, min_x:0.5:max_x, min_y:0.5:max_y, (x, y) -> hyperplane([x, y]), levels=[-1], linestyles=:dash, colorbar_entry=false, color=:red)
  contour!(plt, min_x:0.5:max_x, min_y:0.5:max_y, (x, y) -> hyperplane([x, y]), levels=[0], linestyles=:solid, label="SVM prediction", colorbar_entry=false, color=:green)
  contour!(plt, min_x:0.5:max_x, min_y:0.5:max_y, (x, y) -> hyperplane([x, y]), levels=[1], linestyles=:dash, colorbar_entry=false, color=:blue)

  display(plt)
end

# show initial hyperplane
draw()
readline()

# uniform randint except `a`
function randint_without(n, a)
  b = rand(1:n-1)
  b += (b >= a)
  return b
end

# performs one step of a simplified SMO
function fit(alpha_tol=0.0001, error_tol=0.0001)
  n = length(α)

  error(k) = hyperplane(X[k]) - Y[k]
  η(i, j) = 2 * k(X[i], X[j]) - k(X[i], X[i]) - k(X[j], X[j])

  LH(i, j) =
    if Y[i] == Y[j]
      (max(0, α[i] + α[j] - C), min(C, α[i] + α[j]))
    else
      (max(0, α[j] - α[i]), min(C, C + α[j] - α[i]))
    end

  for i in 1:n
    j = randint_without(n, i)

    Ei, Ej = error(i), error(j)
    yiEi = Y[i] * Ei

    if (yiEi < -error_tol && α[i] < C) || (yiEi > error_tol && α[i] > 0)
      L, H = LH(i, j)
      if L == H
        continue
      end

      eta = η(i, j)

      if eta >= 0
        continue
      end

      new_j = α[j] - Y[j] * (Ei - Ej) / eta
      new_j = clamp(new_j, L, H)

      if abs(new_j - α[j]) < alpha_tol * (α[j] + new_j + alpha_tol)
        continue
      end

      new_i = α[i] + Y[i] * Y[j] * (α[j] - new_j)

      b1 = b - Ei - Y[i] * (new_i - α[i]) * k(X[i], X[i]) - Y[j] * (new_j - α[j]) * k(X[i], X[j])
      b2 = b - Ej - Y[i] * (new_i - α[i]) * k(X[i], X[j]) - Y[j] * (new_j - α[j]) * k(X[j], X[j])

      α[i] = new_i
      α[j] = new_j
      global b = if α[i] ∈ 0:C
        b1
      elseif α[j] ∈ 0:C
        b2
      else
        (b1 + b2) / 2
      end
    end
  end
end


# train and show the decision boundary
for _ in 1:100
  fit()
end
draw()
