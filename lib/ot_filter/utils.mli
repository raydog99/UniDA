open Torch

val pairwise_distances : Tensor.t -> Tensor.t -> Tensor.t
val sinkhorn_knopp : Tensor.t -> Tensor.t -> Tensor.t -> float -> int -> Tensor.t
val sharpen : Tensor.t -> float -> Tensor.t
val mixup : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t -> float -> Tensor.t * Tensor.t
val add_synthetic_noise : Tensor.t -> float -> bool -> Tensor.t
val accuracy : Tensor.t -> Tensor.t -> float
val augment_image : Tensor.t -> Tensor.t
val augment_batch : Tensor.t -> Tensor.t