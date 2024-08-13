open Torch

val compute_wasserstein_distance : Tensor.t -> Tensor.t -> float -> float
val compute_pairwise_distances : Tensor.t -> Tensor.t -> float -> Tensor.t
val compute_diameter : Tensor.t -> float
val euclidean_distance : Tensor.t -> Tensor.t -> Tensor.t