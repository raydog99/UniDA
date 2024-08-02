open Torch

type regularization =
  | Entropic
  | Quadratic

val sinkhorn_distance :
  Tensor.t -> float -> Tensor.t -> Tensor.t -> regularization ->
  max_iter:int -> tol:float -> Tensor.t

val sinkhorn_divergence :
  Tensor.t -> float -> Tensor.t -> Tensor.t -> regularization ->
  max_iter:int -> tol:float -> Tensor.t

val ot_distance :
  (Tensor.t -> Tensor.t -> Tensor.t) -> float -> Tensor.t -> Tensor.t -> regularization ->
  max_iter:int -> tol:float -> Tensor.t

val sinkhorn_gradient :
  (Tensor.t -> Tensor.t -> Tensor.t) -> float -> Tensor.t -> Tensor.t -> regularization ->
  max_iter:int -> tol:float -> Tensor.t * Tensor.t * Tensor.t

val normalize : Tensor.t -> dim:int -> Tensor.t

val create_histogram : Tensor.t -> int -> Tensor.t