open Torch

module type Config = sig
  val epsilon : float
  val beta : float
  val alpha : float
  val t_max : int
  val j_max : int
  val n_samples : int
  val n_features : int
end

module Make (C : Config) : sig
  type t

  val create : Tensor.t -> t
  val optimize : t -> Tensor.t -> Tensor.t
end