open Torch

type t

val create : Tensor.t -> Tensor.t -> int -> t
val batch : t -> int -> Tensor.t * Tensor.t
val get_all : t -> Tensor.t * Tensor.t