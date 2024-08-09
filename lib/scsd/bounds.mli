open Torch

val approximation_error_bound : float -> int -> float -> float -> float
val sample_complexity_bound : float -> int -> float -> float -> int -> float -> float
val estimate_lipschitz_constant : Tensor.t -> float
val estimate_diameter : Tensor.t -> float