open Torch

module type S = sig
  val estimate : Tensor.t -> Tensor.t -> int -> int -> float -> int -> Tensor.t
  val create_blocks : Tensor.t -> int -> Tensor.t list
  val median_block : Tensor.t list -> Tensor.t
end

val create_blocks : Tensor.t -> int -> Tensor.t list
val median_block : Tensor.t list -> Tensor.t
val clip_weights : Tensor.t -> float -> Tensor.t
val estimate : Tensor.t -> Tensor.t -> int -> int -> float -> int -> Tensor.t