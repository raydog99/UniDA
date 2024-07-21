open Torch

val compute_euclidean_cost_matrix : Tensor.t -> Tensor.t
val compute_gromov_cost_matrix : Tensor.t -> Tensor.t
val solve_linear_program : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t
val sinkhorn : Tensor.t -> Tensor.t -> Tensor.t -> float -> int -> Tensor.t