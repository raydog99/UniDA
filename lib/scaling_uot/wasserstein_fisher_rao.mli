open Torch

val wfr_cost : Tensor.t -> float -> Tensor.t
(** Wasserstein-Fisher-Rao cost function *)

val wfr_proximal : Tensor.t -> Tensor.t -> float -> Tensor.t
(** Wasserstein-Fisher-Rao proximal operator *)

val wfr_distance : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t -> float -> float
(** Wasserstein-Fisher-Rao distance *)