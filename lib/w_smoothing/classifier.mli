open Torch

val create : unit -> Tensor.t -> Tensor.t
val train : (Tensor.t -> Tensor.t) -> Tensor.t -> Tensor.t -> unit
val evaluate : (Tensor.t -> Tensor.t) -> Tensor.t -> Tensor.t -> float