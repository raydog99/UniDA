open Torch

val compute_prototypes : Tensor.t -> Tensor.t -> int -> Tensor.t
val update_prototypes : Tensor.t -> Tensor.t -> float -> Tensor.t