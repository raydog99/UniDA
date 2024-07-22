open Torch
open Regularization

type cost_matrix = Tensor.t
type transport_plan = Tensor.t
type marginals = Tensor.t * Tensor.t

val delta_omega : Regularization.t -> Tensor.t -> Tensor.t
val grad_delta_omega : Regularization.t -> Tensor.t -> Tensor.t
val max_omega : Regularization.t -> Tensor.t -> Tensor.t
val grad_max_omega : Regularization.t -> Tensor.t -> Tensor.t
val ot_omega : Tensor.t -> Tensor.t -> Tensor.t -> Regularization.t -> transport_plan
val smoothed_semi_dual : Tensor.t -> Tensor.t -> Tensor.t -> Regularization.t -> Tensor.t