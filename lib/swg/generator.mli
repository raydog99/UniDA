open Torch

type t

val create : int -> int -> t
val forward : t -> Tensor.t -> Tensor.t