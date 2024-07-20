open Torch

include Estimator.S

val create_blocks : Tensor.t -> int -> Tensor.t list
val median_block : Tensor.t list -> Tensor.t