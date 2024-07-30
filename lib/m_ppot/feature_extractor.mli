open Torch

val create : unit -> Nn.t
val extract_features : Nn.t -> Tensor.t -> Tensor.t