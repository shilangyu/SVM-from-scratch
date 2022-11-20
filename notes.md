# Support Vector Machines (SVM)

> Disclaimer: this does not aim to fully cover the possibilities of SVM models. It merely describes the basic concepts related to them. Some details are skipped on purpose with the intention of keeping it short.

Invented by Vladimir Vapnik at Bell Labs (yes, this Bell Labs). SVM is a binary linear classifier for supervised learning. Input data are points in Euclidean space.

Let $D = \{(x_i, y_i) : i \in \{1, \cdots, n\}\}$ be a dataset which is a set of pairs where $x_i \in \mathbb R^d$ is a _data point_ in some $d$-dimensional space and $y_i \in \{-1, 1\}$ is a _label_ of the appropriate $x_i$ data point classifying it to one of the two classes. The model is trained on $D$ after which it is present with $x_{i+1}$ and is asked to predict the label of this previously unseen data point.

The prediction function will be denoted by $p: \mathbb R^d \to \{-1, 1\}$. The output of a prediction will be denoted by $\hat y$. SVM is a description of such a model and how can one optimize $p$ given a dataset and a loss function.

SVM's goal is to construct a prediction function which will represent a hyperplane that can be used to divide the space into two parts. One SVM model is considered to be better than a different SVM model for the same dataset if the margin (distance) between the hyperplane and the nearest data point is maximized. The nearest data point to the hyperplane is called the _support vector_. Therefore we have a clear metric to optimize.

<!-- ![A dataset with black and white dots representing two different labels. Three hyperplanes divide the dataset in different ways.](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Svm_separating_hyperplanes_%28SVG%29.svg/1920px-Svm_separating_hyperplanes_%28SVG%29.svg.png) -->

Recall the general equation of a hyperplane: $w \cdot x - b = 0$ where $w \in \mathbb R^d$ denotes a normal vector to the hyperplane and $b \in \mathbb R$ is the offset ($\frac{b}{||w||}$ determines the offset from the origin along the normal vector $w$). Since our goal is the find the optimal hyperplane, we end up with $d+1$ trainable parameters ($|w| + 1$). Once the hyperplane is found we can construct two additional parallel hyperplanes which reside at the support vectors of the two classes, $w \cdot x - b = -1$ and $w \cdot x - b = 1$. Then, all points from the dataset adhere to the following

$$
y_i = \begin{cases}
	-1 & \text{if } w \cdot x_i - b \le -1 \\
	1 & \text{if } w \cdot x_i - b \ge 1 \\
\end{cases} \implies y_i(w \cdot x - b) \ge 1
$$

<!-- ![A dataset separated by a hyperplane normalized around the support vectors.](https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/SVM_margin.png/1920px-SVM_margin.png) -->

Since $\frac{1}{||w||}$ is the margin and we want to maximize it, the problem can be restated as a minimization problem of $||w||$. Our predictor can be neatly expressed as $p(x) = \text{sign}(w \cdot x - b)$ with an edge case of when $x$ lies perfectly on the hyperplane. This is called a _hard-margin SVM_ since it works only for perfect datasets which do not have outliers.

Now that we have the model we need to introduce a way to train it. There are many techniques to do so. Here we will focus on one which uses gradient descent. Firstly, we need some function we want to optimize. We will use the hinge function which will suit our needs well: $H(x_i, y_i) = \max(0, 1 - y_i(w \cdot x_i - b))$. Notice, that when the guess is correct, then $y_i(w \cdot x_i - b) \ge 1$ as shown before, thus $H = 0$. If the guess is incorrect, $H \ge 0$. So if for every data point $H = 0$ then we have found a hyperplane the space correctly. Finally, we can define a loss function over the whole dataset which we will want to minimize:

$$
\ell(w, b) = \frac{1}{n}\sum_{i=1}^n H(x_i, y_i) = \frac{1}{n}\sum_{i=1}^n \max(0, 1 - y_i(w \cdot x_i - b))
$$

To perform gradient descent we will need to compute the partial derivatives with respect to the trainable parameters ($w$ and $b$). Let's start by considering the hinge loss function which can be split into two cases: when we reach the left and right case of the $\max$ function.

$$
H(x_i, y_i) = \begin{cases}
	0 & \text{if } y_i(w \cdot x_i - b) \ge 1 \\
	1 - y_i(w \cdot x_i - b) & \text{otherwise} \\
\end{cases}
$$

Which yields the following derivatives (recall that $w$ is a vector)

$$
\frac{\partial H_i}{\partial w} = \begin{cases}
	0 & \text{if } y_i(w \cdot x_i - b) \ge 1 \\
	-y_i x_i & \text{otherwise} \\
\end{cases}
$$

$$
\frac{\partial H_i}{\partial b} = \begin{cases}
	0 & \text{if } y_i(w \cdot x_i - b) \ge 1 \\
	y_i & \text{otherwise} \\
\end{cases}
$$

> See [linear_hard_margin_svm.jl](linear_hard_margin_svm.jl) for a practical implementation of the so far introduced concepts

## Problem 1: what if the dataset isn't perfect?

TODO

## Problem 2: what if the problem is not linearly separable?

TODO

## Problem 3: what if the problem isn't binary?

TODO

## References

1. All images taken from the well-written Wikipedia article on SVMs <https://en.wikipedia.org/wiki/Support_vector_machine>
