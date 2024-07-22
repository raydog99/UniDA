open Torch

val transform : Tensor.t -> (Tensor.t -> Tensor.t -> Tensor.t) -> Tensor.t -> Tensor.t