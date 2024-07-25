open Torch

val frank_wolfe_step : Tensor.t -> int -> Tensor.t
val frank_wolfe_algorithm : Tensor.t -> Tensor.t -> int -> float -> int -> float -> Tensor.t