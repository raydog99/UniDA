open Torch

type t

val create : int -> int -> t
val forward : t -> Tensor.t -> Tensor.t
val predict : t -> Tensor.t -> Tensor.t
val supervised_loss : t -> Tensor.t -> Tensor.t -> Tensor.t
val unsupervised_loss : t -> Tensor.t -> Tensor.t -> Tensor.t