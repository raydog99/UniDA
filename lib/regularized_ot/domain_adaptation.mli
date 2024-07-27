open Torch

val adapt :
  Tensor.t -> Tensor.t -> float -> int -> float -> Tensor.t
(** Perform domain adaptation using regularized optimal transport
    @param source_features Source domain features
    @param target_features Target domain features
    @param lambda Regularization parameter
    @param max_iter Maximum number of iterations for Sinkhorn algorithm
    @param epsilon Convergence threshold for Sinkhorn algorithm
    @return Adapted source features
*)

val adapt_default :
  Tensor.t -> Tensor.t -> Tensor.t
(** Perform domain adaptation using regularized optimal transport with default parameters
    @param source_features Source domain features
    @param target_features Target domain features
    @return Adapted source features
*)

val adapt_with_transport_plan :
  Tensor.t -> Tensor.t -> float -> int -> float -> 
  Tensor.t * Tensor.t
(** Perform domain adaptation and return both adapted features and transport plan
    @param source_features Source domain features
    @param target_features Target domain features
    @param lambda Regularization parameter
    @param max_iter Maximum number of iterations for Sinkhorn algorithm
    @param epsilon Convergence threshold for Sinkhorn algorithm
    @return Tuple of (adapted_features, transport_plan)
*)