open Torch

exception Invalid_input of string

val sgd_semi_discrete_ot : 
  (unit -> Tensor.t) -> Tensor.t -> (Tensor.t -> Tensor.t) -> float -> int -> float -> Tensor.t

val minibatch_sgd_semi_discrete_ot : 
  (int -> Tensor.t) -> Tensor.t -> (Tensor.t -> Tensor.t) -> float -> int -> float -> int -> Tensor.t

val h_epsilon : Tensor.t -> Tensor.t -> Tensor.t -> (Tensor.t -> Tensor.t) -> float -> Tensor.t