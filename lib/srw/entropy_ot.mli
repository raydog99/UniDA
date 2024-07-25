open Torch

val sinkhorn : Tensor.t -> Tensor.t -> Tensor.t -> float -> int -> Tensor.t
val entropy_regularized_ot : Tensor.t -> Tensor.t -> (Tensor.t -> Tensor.t -> Tensor.t) -> float -> int -> Tensor.t