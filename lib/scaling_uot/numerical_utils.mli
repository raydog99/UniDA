open Torch

val stabilize_tensor : Tensor.t -> float -> Tensor.t
(** Stabilize a tensor by setting a minimum value *)

val log_sum_exp : Tensor.t -> Tensor.t
(** Compute log(sum(exp(x))) in a numerically stable way *)

val safe_div : Tensor.t -> Tensor.t -> float -> Tensor.t
(** Perform division with a small epsilon to avoid division by zero *)