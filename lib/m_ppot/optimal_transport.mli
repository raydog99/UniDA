open Torch

val compute_cost_matrix : Tensor.t -> Tensor.t -> Tensor.t
val sinkhorn : Tensor.t -> float -> int -> float -> Tensor.t
val solve_pot : Tensor.t -> float -> Tensor.t