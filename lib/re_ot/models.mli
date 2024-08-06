open Torch

type t

val create : int -> int -> int -> t
val forward : t -> Tensor.t -> Tensor.t
val parameters : t -> NN.Parameters.t
val update_q : t -> Tensor.t -> unit