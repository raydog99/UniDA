open Torch

exception DimensionMismatch of string

val kl_div : Tensor.t -> Tensor.t -> Tensor.t
(** Compute KL divergence between two tensors *)

val entropy : Tensor.t -> Tensor.t
(** Compute entropy of a tensor *)

val sinkhorn : ?num_iterations:int -> ?epsilon:float -> ?tol:float -> 
               Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t
(** Sinkhorn algorithm for optimal transport *)

val compute_cost : Tensor.t -> Tensor.t -> Tensor.t
(** Compute cost matrix between two sets of points *)

val normalize_tensor : Tensor.t -> Tensor.t
(** Normalize a tensor to sum to 1 *)

val check_probability_vector : Tensor.t -> bool
(** Check if a tensor is a valid probability vector *)