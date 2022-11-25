---
linkcolor: cyan
---

# Support Vector Machines (SVM)

> Disclaimer: this does not aim to fully cover the possibilities of SVM models. It merely describes the basic concepts related to them. Some details are skipped on purpose with the intention of keeping it short.

Invented by Vladimir Vapnik. SVM is a binary linear classifier for supervised learning (though, can be used for regression as well). Input data are points in Euclidean space.

Let $D = \{(x_i, y_i) : i \in \{1, \cdots, n\}\}$ be a dataset which is a set of pairs where $x_i \in \mathbb R^d$ is a _data point_ in some $d$-dimensional space and $y_i \in \{-1, 1\}$ is a _label_ of the appropriate $x_i$ data point classifying it to one of the two classes. The model is trained on $D$ after which it is present with $x_{i+1}$ and is asked to predict the label of this previously unseen data point.

The prediction function will be denoted by $p: \mathbb R^d \to \{-1, 1\}$. The output of a prediction will be denoted by $\hat y$. SVM is a description of such a model and how can one optimize $p$ given a dataset and a loss function.

SVM's goal is to construct a prediction function which will represent a hyperplane that can be used to divide the space into two parts. One SVM model is considered to be better than a different SVM model for the same dataset if the margin (distance) between the hyperplane and the nearest data point is maximized. The nearest data point to the hyperplane is called the _support vector_. Therefore we have a clear metric to optimize.

![A dataset with black and white dots representing two different labels. Three hyperplanes divide the dataset in different ways.](<https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Svm_separating_hyperplanes_(SVG).svg/1920px-Svm_separating_hyperplanes_(SVG).svg.png>)

Recall the general equation of a hyperplane: $w \cdot x - b = 0$ where $w \in \mathbb R^d$ denotes a normal vector to the hyperplane and $b \in \mathbb R$ is the offset ($\frac{b}{||w||}$ determines the offset from the origin along the normal vector $w$). Since our goal is the find the optimal hyperplane, we end up with $d+1$ trainable parameters ($|w| + 1$). Once the hyperplane is found we can construct two additional parallel hyperplanes which reside at the support vectors of the two classes, $w \cdot x - b = -1$ and $w \cdot x - b = 1$. Then, all points from the dataset adhere to the following

$$
y_i = \begin{cases}
	-1 & \text{if } w \cdot x_i - b \le -1 \\
	1 & \text{if } w \cdot x_i - b \ge 1 \\
\end{cases} \implies y_i(w \cdot x - b) \ge 1
$$

![A dataset separated by a hyperplane normalized around the support vectors.](https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/SVM_margin.png/1920px-SVM_margin.png)

Since $\frac{1}{||w||}$ is the margin and we want to maximize it, the problem can be restated as a minimization problem of $||w||$. Our predictor can be neatly expressed as $p(x) = \text{sign}(w \cdot x - b)$ with an edge case of when $x$ lies perfectly on the hyperplane, then we can just assume that $p(x) = 0$ belongs to one of the two classes. This is called a _hard-margin SVM_ since it works only for perfect datasets which do not have outliers.

Now that we have the model we need to introduce a way to train it. There are many techniques to do so. Here we will focus on one which uses gradient descent. Firstly, we need some function we want to optimize. We will use the hinge function which will suit our needs well: $H(x_i, y_i) = \max(0, 1 - y_i(w \cdot x_i - b))$. Notice, that when the guess is correct, then $y_i(w \cdot x_i - b) \ge 1$ as shown before, thus $H = 0$. If the guess is incorrect, $H \ge 0$. So if for every data point $H = 0$ then we have found a hyperplane that divides the space correctly. Hinge loss introduces a _soft-margin_ since it allows for misclassification with a quantifiable result. We also have to incorporate the minimization of $||w||$ as previously stated. Finally, we can define a loss function over the whole dataset which we will want to minimize:

$$
\begin{aligned}
	\ell(w, b) &= \lambda ||w||^2 + \frac{1}{n}\sum_{i=1}^n H(x_i, y_i) \\
	 					 &= \lambda w \cdot w + \frac{1}{n}\sum_{i=1}^n \max(0, 1 - y_i(w \cdot x_i - b))
\end{aligned}
$$

Here $\lambda > 0$ is the regularization hyperparameter controlling the trade-off between correct predictions and large margins. To perform gradient descent we will need to compute the partial derivatives with respect to the trainable parameters ($w$ and $b$). Sadly the hinge function is not differentiable, but we can consider it by splitting it into two cases: when we reach the left and right case of the $\max$ function.

$$
H(x_i, y_i) = \begin{cases}
	0 & \text{if } y_i(w \cdot x_i - b) \ge 1 \\
	1 - y_i(w \cdot x_i - b) & \text{otherwise} \\
\end{cases}
$$

Which yields the following gradient (recall that $w$ is a vector):

$$
\nabla\ell = \begin{bmatrix}
	\frac{\partial \ell}{\partial w} \\
	\\
	\frac{\partial \ell}{\partial b} \\
\end{bmatrix} = \begin{bmatrix}
	2\lambda w + \frac{1}{n}\sum_{i = 0}^n \begin{cases}
		0 & \text{if } y_i(w \cdot x_i - b) \ge 1 \\
		-y_i x_i & \text{otherwise} \\
	\end{cases} \\
	\frac{1}{n}\sum_{i = 0}^n \begin{cases}
		0 & \text{if } y_i(w \cdot x_i - b) \ge 1 \\
		y_i & \text{otherwise} \\
	\end{cases} \\
\end{bmatrix} = \mathbf 0
$$

For each training example from our dataset we can now first check the $y_i(w \cdot x_i - b) \ge 1$ condition. We can perform gradient descent with the gradient specified above and conditionally apply a different gradient based on the condition. Since the gradient points to the steepest ascent and our task is to minimize the function, we will subtract the gradient instead of adding it. Our parameters will now converge iteratively, where $k$ is the iteration number for each $i$:

$$
w_{k+1} = \begin{cases}
	w_k - 2\lambda w &\text{if } y_i(w \cdot x_i - b) \ge 1 \\
	w_k - (2\lambda w - y_i x_i) = w_k - 2\lambda w + y_i x_i &\text{otherwise} \\
\end{cases} \\
$$

$$
b_{k+1} = \begin{cases}
	b_k &\text{if } y_i(w \cdot x_i - b) \ge 1 \\
	b_k - y_i &\text{otherwise} \\
\end{cases} \\
$$

> See [linear_soft_margin_svm.jl](linear_soft_margin_svm.jl) for a practical implementation of the so far introduced concepts

## What if the problem is not linearly separable?

So far we have been generating hyperplanes which intrinsically suffer from being able to classify only linearly separable datasets. For instance, the XOR function is not linearly separable. To solve non-linear problems one has to introduce non-linearity to the model, similarly to how neural networks use non-linear activation functions. One such way would be to introduce extra dimensions where the dataset can be divided by a hyperplane. If $\mathbb R^d$ is the space of the data points, we want to find such a mapping, called feature map, $\varphi: \mathbb R^d \to \mathbb R^r$ for some $r \in \mathbb N$ (preferably $d < r$ since we want to increase the dimensionality) together with a _kernel function_ $k: \mathbb R^d \times \mathbb R^d \to \mathbb R$

$$
	k(x, y) = \langle \varphi(x), \varphi(y) \rangle
$$

Since we are operating in the euclidean space with the dot product as the inner product space, we can rewrite $k$ as $k(x,y) = \varphi(x) \cdot \varphi(y)$. Direct computation of $\varphi$ is not needed, we will only need to find the kernel function. This kernel function will replace dot products used throughout the SVM method.

![A non-linearly separable dataset mapped to a higher dimensional space where a decision boundary can be found with a simple hyperplane using $\varphi(X) = (x_1, x_2, x_1^2 + x_2^2)$.](https://upload.wikimedia.org/wikipedia/commons/thumb/c/cc/Kernel_trick_idea.svg/440px-Kernel_trick_idea.svg.png)

Note that if we set $\varphi = \text{id}$ then $k(x, y) = \langle \varphi(x), \varphi(y) \rangle = \langle x, y \rangle = x \cdot y$, which gives us the SVM for linearly separable problems. Now, this problem can be expressed as a primal one and proceed with gradient descent as we did for the linear case. However, for some not precisely known reason the literature about SVMs has converged towards solving it as a Lagrangian dual and, as O. Chapelle (2007) argues, the primal one can provide many advantages. In both the primal and dual problem formulation $\varphi$ is never directly computed, it is always used indirectly through the kernel function. So, to not diverge from literature I will present the dual problem which requires Quadratic Optimization instead of gradient descent.

To start we introduce the Lagrange primal which we want to minimize on $w$ and $b$

$$
\mathcal L_1(w, b, \alpha) = \lambda\frac{1}{2}||w||^2 - \sum_{i=1}^n \alpha_i (y_i(w^T\varphi(x_i) - b) - 1)
$$

subject to $\alpha_i \ge 0$. This is not enough since we want to get rid of the $\varphi$. Thus we attempt to minimize on $w$ and $b$ to derive the dual:

$$
	\frac{\partial \mathcal L_1}{\partial w} = 0 = \lambda w - \sum_{i=1}^n \alpha_i y_i \varphi(x_i) \implies w = \sum_{i=1}^n \alpha_i y_i \varphi(x_i)
$$

$$
	\frac{\partial \mathcal L_1}{\partial b} = 0 = -\sum_{i=1}^n \alpha_i y_i \implies \sum_{i=1}^n \alpha_i y_i = 0
$$

Which yields the dual which has to be minimized as

$$
\begin{aligned}
	\mathcal L_2(\alpha) &= \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n y_i \alpha_i k(x_i, x_j) y_j \alpha_j - \sum_{i=1}^n \alpha_i \\
	 &= \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n y_i \alpha_i k(x_i, x_j) y_j \alpha_j - \sum_{i=1}^n \alpha_i \\
\end{aligned}
$$

Subject to $\sum_{i=1}^n \alpha_i y_i = 0$ and $0 \le \alpha_i \le \lambda^{-1}$. To solve this one has to use quadratic optimization which is well beyond the scope of this document. A popular approach is the SMO algorithm (J. Platt 1998).

In the new space $w$ can be expressed as the following linear combination, for $c$ support vectors (ie. those with $\alpha_i > 0$):

$$
	w = \sum_{i=1}^c \alpha_i y_i \varphi(x_i)
$$

And $b$ can be computed from the support vectors as well:

$$
	b = \sum_{i=1}^c \alpha_i y_i \varphi(x_i) - y_i
$$

Then we can return to our predictor function

$$
\begin{aligned}
	p(x) &= \text{sign}(w \cdot \varphi(x) - b) \\
			 &= \text{sign}(\sum_{i=1}^c \alpha_i y_i \varphi(x_i) \cdot \varphi(x) - b) \\
			 &= \text{sign}(\sum_{i=1}^c \alpha_i y_i \langle \varphi(x_i), \varphi(x) \rangle - b) \\
			 &= \text{sign}(\sum_{i=1}^c \alpha_i y_i k(x_i, x) - b) \\
\end{aligned}
$$

Many kernel function exist and can be used for different use cases. A common choice for a kernel function is the basis radial function, it has a single hyperparameter $\gamma > 0$:

$$
	\text{rbf}(x, y) = \exp(-\gamma||x - y||^2)
$$

> See [non_linear_svm.jl](non_linear_svm.jl) for a practical implementation of a SVM with a $\text{rbf}$ kernel trained on a non linearly separable dataset.

## What if the problem isn't binary?

If the amount of classes is larger than 2, we can construct multiple SVM and treat them as a single larger SVM. There are many popular techniques for that, but here two most popular approaches will be mentioned. Let there be $m$ classes.

1. _one-versus-all_: we construct $m$ SVMs trained to treat the dataset as having two classes: one for the target class, and the other for all other $m-1$ classes. To then perform predictions, we can run the new $x_{i+1}$ point through all $m$ SVMs and see which one is the most certain about its prediction. Note, that the definition of the prediction function had a co-domain of $\{-1, 0, 1\}$ so it is not possible to decide which SVM is the most certain. Therefore the prediction function has to be altered to produce quantifiable scores.
2. _one-versus-one_: we construct $\binom{m}{2}$ SVMs for every combination of pairs of classes. Then to perform predictions, we run all SVMs and collect votes. The class with most votes wins.

In the case of _one-versus-all_ the prediction function has to be reformulated unlike in the _one-versus-one_ case. However, _one-versus-one_ comes with a $\binom{m}{2} = \mathcal O(m^2)$ quadratic amount of SVMs unlike the $m = \mathcal O(m)$ linear one for _one-versus-all_. Thus the _one-versus-one_ approach will scale horribly for larger values of $m$.

> See [multiclass_svm.jl](multiclass_svm.jl) for a practical implementation of a multiclass SVM using the _one-versus-all_ approach.

## References

1. All images taken from the well-written Wikipedia article on SVMs [https://en.wikipedia.org/wiki/Support_vector_machine](https://en.wikipedia.org/wiki/Support_vector_machine)
2. **V. Vapnik**, _Statistical Learning Theory_ (1998)
3. **O. Chapelle**, _Training a Support Vector Machine in the Primal_ (2007)
4. **J. Platt**, _Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines_ (1998)
