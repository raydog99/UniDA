open Torch

val project_onto_simplex : Tensor.t -> Tensor.t
(** Project a tensor onto the probability simplex *)

val project_onto_cube : Tensor.t -> Tensor.t
(** Project a tensor onto the [-1, 1] cube *)