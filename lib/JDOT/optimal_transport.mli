open Torch

val sinkhorn_knopp : Tensor.t -> float -> int -> Tensor.t
val compute_cost_matrix : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t -> float -> Tensor.t