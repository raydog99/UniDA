open Torch

type t

val create : int -> int -> int -> t
val train : t -> Tensor.t -> int -> int -> int -> float -> t
val generate : t -> int -> Tensor.t