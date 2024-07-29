open Torch

type rejection_result = {
  pi: Tensor.t;
  mu_t: Tensor.t;
  rejected_indices: Tensor.t;
}

val rejection : ?epsilon:float -> Tensor.t -> Tensor.t -> Tensor.t -> float -> rejection_result
(** Perform rejection step *)

val apply_rejection : rejection_result -> Tensor.t -> Tensor.t
(** Apply rejection to target samples *)

val get_accepted_indices : rejection_result -> Tensor.t
(** Get indices of accepted samples *)