open Torch

type t

val create : int -> int -> int -> (t, string) result
val forward : t -> Tensor.t -> (Tensor.t array, string) result
val parameters : t -> Tensor.t list