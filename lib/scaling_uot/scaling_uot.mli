open Torch
open Divergences

type cost = Tensor.t
type kernel = Tensor.t
type scaling = Tensor.t
type marginal = Tensor.t

type error =
  | InvalidDimension
  | NegativeValues
  | DivergenceError of string
  | MaxIterationsReached
  | NumericalInstability

val compute_kernel : cost -> float -> kernel
(** Compute kernel from cost matrix *)

val scaling_iteration : 
  kernel -> scaling -> scaling -> marginal -> marginal -> 
  float -> divergence -> divergence -> (scaling * scaling, error) result
(** Perform a single scaling iteration *)

val scaling_algorithm : 
  cost -> marginal -> marginal -> float -> int -> float -> 
  divergence -> divergence -> (scaling * scaling, error) result
(** Main scaling algorithm for unbalanced optimal transport *)

val compute_transport_plan : scaling -> kernel -> scaling -> Tensor.t
(** Compute optimal transport plan *)

val wasserstein_distance : cost -> Tensor.t -> float
(** Compute Wasserstein distance *)

val wasserstein_fisher_rao_distance : 
  marginal -> marginal -> cost -> float -> (float, error) result
(** Compute Wasserstein-Fisher-Rao distance *)

val gaussian_hellinger_kantorovich_distance : 
  marginal -> marginal -> cost -> float -> float -> (float, error) result
(** Compute Gaussian-Hellinger-Kantorovich distance *)

val entropy_regularized_ot : 
  cost -> marginal -> marginal -> float -> (Tensor.t, error) result
(** Compute entropy-regularized optimal transport plan *)

val print_transport_plan : Tensor.t -> unit
(** Print the transport plan *)

val print_distance : float -> unit
(** Print the computed distance *)

val example_usage : unit -> unit
(** Example usage of the scaling algorithm *)

val wfr_example : unit -> unit
(** Example usage of the Wasserstein-Fisher-Rao distance *)

val ghk_example : unit -> unit
(** Example usage of the Gaussian-Hellinger-Kantorovich distance *)