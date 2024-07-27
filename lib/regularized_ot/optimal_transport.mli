open Torch

val compute_transport_plan :
  Tensor.t -> Tensor.t -> float -> int -> float -> Tensor.t
(** Compute the regularized optimal transport plan
    @param source_features Source domain features
    @param target_features Target domain features
    @param lambda Regularization parameter
    @param max_iter Maximum number of iterations for Sinkhorn algorithm
    @param epsilon Convergence threshold for Sinkhorn algorithm
    @return Optimal transport plan as a tensor
*)

val compute_transport_plan_default :
  Tensor.t -> Tensor.t -> Tensor.t
(** Compute the regularized optimal transport plan with default parameters
    @param source_features Source domain features
    @param target_features Target domain features
    @return Optimal transport plan as a tensor
*)  