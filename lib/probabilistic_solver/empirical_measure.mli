open Torch

val sample_empirical_measure : Tensor.t -> int -> Tensor.t
val create_measure_from_points : Tensor.t -> Tensor.t
val normalize_measure : Tensor.t -> Tensor.t
val histogram_to_measure : Tensor.t -> Tensor.t