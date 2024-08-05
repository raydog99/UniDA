open Torch

type t

val create : int -> int -> t
val add : t -> Tensor.t -> unit
val get : t -> Tensor.t