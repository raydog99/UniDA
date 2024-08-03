open Torch

module type Cost_fn = sig
  val cost : Tensor.t -> Tensor.t -> Tensor.t
end

module OT : functor (C : Cost_fn) -> sig
  val primal_problem : Tensor.t -> Tensor.t -> Tensor.t -> float -> float
  val dual_problem : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t -> float -> float
  val semi_dual_problem : Tensor.t -> Tensor.t -> Tensor.t -> float -> float

  val adaptive_sag_discrete_ot : Tensor.t -> Tensor.t -> Tensor.t -> float -> int -> Tensor.t
  val minibatch_sgd_semi_discrete_ot : 
    (int -> Tensor.t) -> Tensor.t -> (Tensor.t -> Tensor.t) -> float -> int -> float -> int -> Tensor.t
end

module Squared_euclidean_cost : Cost_fn

module OT_squared_euclidean : sig
  include module type of OT(Squared_euclidean_cost)
end