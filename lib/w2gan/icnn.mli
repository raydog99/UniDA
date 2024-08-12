open Torch

type t

val create : int -> int list -> int -> t
val forward : t -> Tensor.t -> Tensor.t
val gradient : t -> Tensor.t -> Tensor.t
val parameters : t -> Layer.t list
val input_dim : t -> int
val output_dim : t -> int
val state_dict : t -> (string * Layer.t) list
val load : (string * Layer.t) list -> t