open Torch

val generate_dummy_data : int -> int -> int -> int -> (Tensor.t * Tensor.t) list
val generate_dummy_target_data : int -> int -> int -> Tensor.t list