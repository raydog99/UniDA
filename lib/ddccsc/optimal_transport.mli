open Torch

val sinkhorn : Tensor.t -> float -> int -> Tensor.t
val ot_loss : Tensor.t -> Tensor.t -> float -> int -> Tensor.t