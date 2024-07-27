open Torch

val algorithm : 
  Tensor.t -> Tensor.t -> Tensor.t -> 
  float -> int -> float -> Tensor.t * Tensor.t
(** Sinkhorn algorithm for computing optimal transport plan
    @param source_dist Source distribution
    @param target_dist Target distribution
    @param cost_matrix Cost matrix between source and target
    @param lambda Regularization parameter
    @param max_iter Maximum number of iterations
    @param epsilon Convergence threshold
    @return Tuple of (u, v) vectors for the optimal transport plan
*)