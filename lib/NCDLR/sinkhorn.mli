open Types
open Buffer
open Torch

val sinkhorn_knopp : Tensor.t -> Tensor.t -> Tensor.t -> int -> float -> Tensor.t
val adaptive_self_labeling_loss : Tensor.t -> Tensor.t -> Tensor.t -> float -> t -> Tensor.t