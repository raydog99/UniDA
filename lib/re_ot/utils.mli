open Torch

val softmax : Tensor.t -> int -> Tensor.t
val kl_divergence : Tensor.t -> Tensor.t -> Tensor.t
val cosine_distance : Tensor.t -> Tensor.t -> Tensor.t