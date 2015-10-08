##BCorrCA Bayesian Correlated Component Analysis
[A,Z] = BCorrCA(X,opts) computes the correlated components, Z, and
their corresponding forward models, A, for the multiview input, X, as
described in [1]. The model estimates a shared forward model, U, and
estimates the similarity between it and A. The variable names follows 
the notation used in [1].

[1] Kamronn, S., Poulsen, A. T., & Hansen, L. K. (2015). Multiview
Bayesian Correlated Component Analysis. Neural Computation.