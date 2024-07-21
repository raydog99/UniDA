open Torch

type cost_matrix = Tensor.t

val partial_wasserstein : 
cost_matrix -> Tensor.t -> Tensor.t -> float -> Tensor.t

val partial_gromov_wasserstein : 
cost_matrix -> cost_matrix -> Tensor.t -> Tensor.t -> float -> Tensor.t

val extend_cost_matrix : cost_matrix -> float -> float -> cost_matrix

val compute_partial_gw_loss : cost_matrix -> cost_matrix -> Tensor.t -> float

val gradient_partial_gw : cost_matrix -> cost_matrix -> Tensor.t -> Tensor.t