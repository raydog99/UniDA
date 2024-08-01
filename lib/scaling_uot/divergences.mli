open Torch

type divergence = Tensor.t -> Tensor.t -> Tensor.t

val kl_divergence : divergence
(** Kullback-Leibler divergence *)

val tv_distance : divergence
(** Total variation distance *)

val range_constraint : float -> float -> divergence
(** Range constraint divergence *)

val proximal_kl : (Tensor.t -> Tensor.t) -> Tensor.t -> float -> Tensor.t
(** Proximal operator for KL divergence *)

val ghk_divergence : float -> divergence
(** Gaussian-Hellinger-Kantorovich divergence *)