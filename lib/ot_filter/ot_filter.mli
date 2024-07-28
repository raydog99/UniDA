open Torch
open Model

val wasserstein_distance : Tensor.t -> Tensor.t -> float -> Tensor.t
val discrete_optimal_transport : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t
val regularized_optimal_transport : Tensor.t -> Tensor.t -> Tensor.t -> float -> Tensor.t
val wasserstein_barycenter : Tensor.t list -> Tensor.t
val find_clean_representations : Tensor.t * Tensor.t -> Tensor.t
val transport_noisy_labels : Tensor.t -> Tensor.t -> Tensor.t
val filter : Tensor.t * Tensor.t -> Tensor.t * Tensor.t
val train : Model.t -> (Tensor.t * Tensor.t) -> int -> int -> float -> Model.t