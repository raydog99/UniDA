open Torch

val smooth : (Tensor.t -> Tensor.t) -> float -> Tensor.t -> Tensor.t
val certify : (Tensor.t -> Tensor.t) -> float -> Tensor.t -> float -> float * int option