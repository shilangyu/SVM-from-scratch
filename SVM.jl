module Models

import LinearAlgebra: ⋅
import Plots: plot
using Plots

export SVM, LinearSVM, fit!, plot, predict

abstract type AbstractSVM end

# SVM with a kernel solving the dual formulation with SMO
mutable struct SVM{Scalar<:Real} <: AbstractSVM
  const C::Scalar
  const kernel

  α::Vector{Scalar}
  b::Scalar

  # training examples in columns
  const X::Matrix{Scalar}
  # all -1 or 1
  const Y::Vector{Int}

  SVM(X::Matrix{Scalar}, Y::Vector{Int}; C::Scalar=one(Scalar), kernel=KernelFunctions.linear) where {Scalar<:Real} = begin
    @assert size(X, 2) == size(Y, 1)
    @assert all(y -> y == -1 || y == 1, Y)

    new{Scalar}(C, kernel, zeros(size(Y, 1)), zero(Scalar), X, Y)
  end
end

(self::SVM)(x) = (self.α .* self.Y) ⋅ self.kernel.(eachcol(self.X), Ref(x)) + self.b

predict(svm::AbstractSVM, x) = x |> sign ∘ svm

function plot(svm::AbstractSVM)
  @assert size(svm.X, 1) == 2 "Plotting the decision boundary works only for 2-dimensional data"

  points1 = svm.X[:, svm.Y.==1]
  points2 = svm.X[:, svm.Y.==-1]

  plt = scatter(points1[1, :], points1[2, :], label="y = 1")
  scatter!(plt, points2[1, :], points2[2, :], label="y = -1")

  min_x, max_x = minimum(svm.X[1, :]), maximum(svm.X[1, :])
  min_y, max_y = minimum(svm.X[2, :]), maximum(svm.X[2, :])

  contour!(plt, min_x:0.5:max_x, min_y:0.5:max_y, (x, y) -> svm([x, y]), levels=[-1], linestyles=:dash, colorbar_entry=false, color=:red)
  contour!(plt, min_x:0.5:max_x, min_y:0.5:max_y, (x, y) -> svm([x, y]), levels=[0], linestyles=:solid, label="SVM prediction", colorbar_entry=false, color=:green)
  contour!(plt, min_x:0.5:max_x, min_y:0.5:max_y, (x, y) -> svm([x, y]), levels=[1], linestyles=:dash, colorbar_entry=false, color=:blue)

  plt
end


# performs one step of a simplified SMO
function fit!(svm::SVM; alpha_tol=0.0001, error_tol=0.0001)
  function randint_without(n, a)
    b = rand(1:n-1)
    b += (b >= a)
    return b
  end

  n = length(svm.α)

  error(k) = svm(svm.X[:, k]) - svm.Y[k]
  η(i, j) = 2 * svm.kernel(svm.X[:, i], svm.X[:, j]) - svm.kernel(svm.X[:, i], svm.X[:, i]) - svm.kernel(svm.X[:, j], svm.X[:, j])

  LH(i, j) =
    if svm.Y[i] == svm.Y[j]
      (max(0, svm.α[i] + svm.α[j] - svm.C), min(svm.C, svm.α[i] + svm.α[j]))
    else
      (max(0, svm.α[j] - svm.α[i]), min(svm.C, svm.C + svm.α[j] - svm.α[i]))
    end

  for i in 1:n
    j = randint_without(n, i)

    Ei, Ej = error(i), error(j)
    yiEi = svm.Y[i] * Ei

    if (yiEi < -error_tol && svm.α[i] < svm.C) || (yiEi > error_tol && svm.α[i] > 0)
      L, H = LH(i, j)
      if L == H
        continue
      end

      eta = η(i, j)

      if eta >= 0
        continue
      end

      new_j = svm.α[j] - svm.Y[j] * (Ei - Ej) / eta
      new_j = clamp(new_j, L, H)

      if abs(new_j - svm.α[j]) < alpha_tol * (svm.α[j] + new_j + alpha_tol)
        continue
      end

      new_i = svm.α[i] + svm.Y[i] * svm.Y[j] * (svm.α[j] - new_j)

      b1 = svm.b - Ei - svm.Y[i] * (new_i - svm.α[i]) * svm.kernel(svm.X[:, i], svm.X[:, i]) - svm.Y[j] * (new_j - svm.α[j]) * svm.kernel(svm.X[:, i], svm.X[:, j])
      b2 = svm.b - Ej - svm.Y[i] * (new_i - svm.α[i]) * svm.kernel(svm.X[:, i], svm.X[:, j]) - svm.Y[j] * (new_j - svm.α[j]) * svm.kernel(svm.X[:, j], svm.X[:, j])

      svm.α[i] = new_i
      svm.α[j] = new_j
      svm.b = if svm.α[i] ∈ 0:svm.C
        b1
      elseif svm.α[j] ∈ 0:svm.C
        b2
      else
        (b1 + b2) / 2
      end
    end
  end
end

# Linear kernel SVM solving the primal formulation (with hinge loss) using gradient descent
# Scales better for a large amount of samples
mutable struct LinearSVM{Scalar<:Real} <: AbstractSVM
  const λ::Scalar

  w::Vector{Scalar}
  b::Scalar

  # training examples in columns
  const X::Matrix{Scalar}
  # all -1 or 1
  const Y::Vector{Int}

  LinearSVM(X::Matrix{Scalar}, Y::Vector{Int}; λ::Scalar=one(Scalar)) where {Scalar<:Real} = begin
    @assert size(X, 2) == size(Y, 1)
    @assert all(y -> y == -1 || y == 1, Y)

    new{Scalar}(λ, zeros(size(X, 1)), zero(Scalar), X, Y)
  end
end

(self::LinearSVM)(x) = self.w ⋅ x - self.b

loss(svm::LinearSVM) = svm.λ * svm.w ⋅ svm.w + sum([max(0, 1 - y * svm(x)) for (x, y) in zip(eachcol(svm.X), svm.Y)]) / length(svm.Y)

# train the model on the dataset. α represents the learning rate
function fit!(svm::LinearSVM; α=0.003)
  for (x, y) in zip(eachcol(svm.X), svm.Y)
    is_correct = y * svm(x) >= 1

    # different gradients have to be applied based on the condition
    if is_correct
      # adjust according to the gradients scaled by the learning rate
      # gradients point to the steepest ascent, but we want to minimize the loss function, so we subtract the gradient
      svm.w -= α * 2svm.λ * svm.w
      svm.b -= α * 0
    else
      svm.w -= α * (2svm.λ * svm.w .- y * x)
      svm.b -= α * y
    end
  end
end

end

module KernelFunctions
import LinearAlgebra: ⋅

linear(x, y) = x ⋅ y

make_rbf(γ::Real) = (x, y) -> exp(-γ * (x - y) ⋅ (x - y))
end
