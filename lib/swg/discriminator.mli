open Torch

type t

val create : int -> t
val forward : t -> Tensor.t -> Tensor.t * Tensor.t
val surrogate_loss : Tensor.t -> Tensor.t -> Tensor.t