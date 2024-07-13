open Torch

module FeatureExtractor : sig
  type t

  val create : int -> int -> t
  val forward : t -> Tensor.t -> Tensor.t
end

module DomainCritic : sig
  type t

  val create : int -> t
  val forward : t -> Tensor.t -> Tensor.t
end

module Discriminator : sig
  type t

  val create : int -> int -> t
  val forward : t -> Tensor.t -> Tensor.t
end