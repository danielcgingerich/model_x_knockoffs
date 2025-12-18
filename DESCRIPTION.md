# Model X knockoffs with gradient boosted machines and neural networks


### Introduction
We present an implementation of the Model-X knockoff framework using flexible, nonparametric machine learning models, including gradient boosted machines and neural networks. The goal is to perform valid feature selection with false discovery rate (FDR) control in high-dimensional settings by testing conditional independence between each predictor and the response.

### Model overview
Letting $X_j$ denote the $j$-th column of $X$, we test for conditional independence of $X_j$ and $Y$, given the other predictor variables, $X_{-j}$:

$$H_{0,j} : X_j \perp Y | X_{-j} $$

The knockoff procedure first characterizes the conditional distribution of $X_j$ given the other predictors. We start by generating a conditional distribution $P(X_j | X_{-j})$, such that the swap property holds:

$$ ( X, 	\tilde{X})_{\text{swap}(S)} \,{\buildrel d \over =}\,  (X, 	\tilde{X}) : 	\tilde{X} \sim P(X_j | X_{-j}) $$

Where $	\tilde{X}$ denotes the knockoff (generated) data, and $X$ denotes the original data matrix. The subscript $\text{swap}(S)$ denotes that we swap columns $S \subset \{1, ..., j \}$ between $X$ and $	\tilde{X}$. 

As $	\tilde{X}$ is constructed irrespective of Y, the predictor-response relationships between $Y$ and the knockoffs are conditionally independent, given the real data. Formal proof is provided in the original work by Barber and Candes. Once this is established, we can run a model to predict $Y$ from the concatenated matrix $X_{KO}=[X, 	\tilde{X}]$:

$$Y = f(X_{KO}) + \varepsilon $$

$$ \varepsilon \sim D_\theta $$

Where $f$ denotes some arbitrary model (LASSO, ridge regression, neural network, etc), $\varepsilon$ denotes the error term, belonging to some distribution, $D$ with parameters $\theta$.  For each feature, we compute a measure of importance, denoted $W_j$ for real features and $	\tilde{W}_j$ for knockoff features. This can be the magnitude of the beta coefficient, feature importance, SHAP values, etc. In our case, we use the SHAP value of each feature for $W$ and $	\tilde{W}$. The test statistic is then computed as

$$V_j = W_j - 	\tilde{W}_j$$

If we assume the null hypothesis that $X_j$ and $Y$ are conditionally independent, then $V_j$ should be summetric about 0. Features with true association should be large and positive. 

Given this information, the distribution of $V_j$'s should be a mixture of null features (symmetric, mean 0, low variance) and true features (large, positive).  The FDR at threshold $t$ can be approximated as:

$$\{FDR}(t) = \frac{
\sum_j \bf{1} \{ V_j \leq -t \}
}{\sum_j \bf{1} \{ V_j \geq t \} }$$

In the above calculation, we take advantage of the fact that null features are symmetric about 0, while alternative features are strictly positive. We can estimate the null proportion by the number of features with $V_j$ less than $-t$, because it is rare that a true feature will take on a negative value.

### Constructing the model X knockoff
The latter procedure is reliant on the fact that we can construct a reliable joint distribution for X. To do this, we apply the sequential conditional independent pairs procedure, outlined in Barber and Candes, 2015. 

*(1) Sampling of the first variable* &mdash; For $j = 1$, we sample conditional $	\tilde{X}_1$, 

$$ 	\tilde{X}_1 | X_{-1} \sim \text{N} ( \mu = f_1 (X_{-1}), \sigma_2 = \exp ( g_1 (X_{-1} ) )$$

Where $f$ and $g$ are arbitrary models describing the conditional mean and variance, respectively. In this repository, we provide model X knockoff frameworks where $f$ and $g$ are gradient boosted machines and also for neural networks. We estimate $f$ and $g$ by optimizing:

$$ f_1 = \text{argmin}_{f_1} \Big( \sum_{i=1}^N \big( X_{i,1} - f_1 (X_{i,-1}) \big)^2 \Big)$$

$$ g_1 = \text{argmin}_{g_1} \Big( \sum_{i=1}^N \big( 
\log (R_1^2) - g_1 (X_{i,-1})) 
\big)^2  \Big)$$

Where $R_1$ is the residual from $f_1$. We predict the log of the squared residuals to enforce that $\hat{\sigma}^2 = \exp(g(X_{-1}))$ is positive. 

*(2) Sequential sampling of remaining variables* &mdash; For $j = 2, ..., p$, we sample the conditional $	\tilde{X}_j$'s: 

$$	\tilde{X}_j | X_{-j}, 	\tilde{X}_{1:(j-1)} \sim \text{N} \big( 
f_j([X_{-j}, X_{1:(j-1)}]), \ g_j([X_{-j}, X_{1:(j-1)}])
\big)$$

Where $f_j$ and $g_j$ are estimated as

$$ f_j = \text{argmin}_{f_j} \Big( 
\sum_{i=1}^N \big( X_{i,j} - f_j (X_{i,-j}, 	\tilde{X}_{i,1:(1-j)}) \big)^2
\Big) $$

$$ g_j = \text{argmin}_{g_j} \Big( 
\sum_{i=1}^N \big( \log R_j^2 - g_j ([X_{i,-j}, 	\tilde{X}_{i,1:(1-j)}])  \big)^2 
\Big) $$

### FDR control of features

We now have a model X knockoff, $X_{KO}=[X, 	\tilde{X}]$, satisfying the swap property. Next, we train a predictive model and calculate a measure of feature importance for each variable of $X_{KO}$. In our case, we use the mean absolute Shapley value for each feature. We calculate the FDR at threshold $t \geq 0$ as: 

$$\text{FDR}(t) = \frac{ (\text{total null features} \geq t)}{ (\text{total features} \geq t)} \approx \frac{ \sum_j \textbf{1}\\{V_j \leq -t\\} }{\sum_j \textbf{1} \\{ V_j \geq t \\} } ;$$

$$ V_j = W_j - 	\tilde{W}_j $$
