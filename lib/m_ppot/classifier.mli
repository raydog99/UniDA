open Torch

val create : int -> int -> Nn.t
val classify : Nn.t -> Tensor.t -> Tensor.t