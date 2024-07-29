open Torch
open Utils

type rejection_result = {
  pi: Tensor.t;
  mu_t: Tensor.t;
  rejected_indices: Tensor.t;
}

let rejection ?(epsilon=1e-3) cost mu_s mu_t threshold =
  if not (check_probability_vector mu_s && check_probability_vector mu_t) then
    raise (Invalid_argument "mu_s and mu_t must be probability vectors")
  else
    let pi = sinkhorn ~epsilon cost mu_s mu_t in
    let mu_t' = Tensor.(sum pi 0) in
    let rejected_indices = Tensor.(lt mu_t' threshold) in
    { pi; mu_t = mu_t'; rejected_indices }

let apply_rejection result x_t =
  let mask = Tensor.(logical_not result.rejected_indices) in
  Tensor.masked_select x_t mask

let get_accepted_indices result =
  Tensor.(nonzero (logical_not result.rejected_indices))