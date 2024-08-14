open Torch

type t

val create : int -> (t, string) result
val forward : t -> Tensor.t -> (Tensor.t, string) result