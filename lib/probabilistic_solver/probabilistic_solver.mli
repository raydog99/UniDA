open Torch

val approximate_wasserstein_distance : Tensor.t -> Tensor.t -> int -> int -> float -> float
val compute_error_bound : int -> int -> float -> int -> float
val compute_covering_number : Tensor.t -> float -> float
val compute_e_q : Tensor.t -> float -> float