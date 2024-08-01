open Torch

val normalize_tensor : Tensor.t -> Tensor.t
(** Normalize a tensor so that it sums to 1 *)

val is_probability_vector : Tensor.t -> bool
(** Check if a tensor is a valid probability vector *)

val generate_random_marginal : int -> Tensor.t
(** Generate a random marginal distribution of given size *)

val generate_random_cost_matrix : int -> int -> Tensor.t
(** Generate a random cost matrix of given dimensions *)

val tensor_to_list : Tensor.t -> float list
(** Convert a 1D tensor to a list of floats *)

val list_to_tensor : float list -> Tensor.t
(** Convert a list of floats to a 1D tensor *)

val print_tensor : Tensor.t -> unit
(** Print a tensor *)