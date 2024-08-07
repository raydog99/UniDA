open Torch

type optimizer = Adam of float | SGD of float
type loss = GAN | CrossEntropy

module type Network = sig
  type t
  val create : int -> int -> int -> t
  val forward : t -> Tensor.t -> Tensor.t
  val parameters : t -> Parameter.t list
end

module type Optimizer = sig
  type t
  val create : Parameter.t list -> float -> t
  val step : t -> unit
  val zero_grad : t -> unit
end

module RobustOT : sig
  val robust_wasserstein_dual :
    Tensor.t -> Tensor.t -> float -> float -> int -> float -> Tensor.t * Tensor.t

  val discrete_formulation :
    Tensor.t -> Tensor.t -> float -> float -> int -> float -> Tensor.t * Tensor.t

  val continuous_relaxation :
    Tensor.t -> Tensor.t -> float -> float -> int -> float -> Tensor.t * Tensor.t

  val gan_optimization :
    (module Network with type t = 'a) ->
    (module Network with type t = 'b) ->
    (module Network with type t = 'c) ->
    (module Optimizer) ->
    'a -> 'b -> 'c ->
    (unit -> Tensor.t) -> float -> float -> int -> float -> unit

  val domain_adaptation_optimization :
    (module Network with type t = 'a) ->
    (module Network with type t = 'b) ->
    (module Network with type t = 'c) ->
    (module Optimizer) ->
    'a -> 'b -> 'c ->
    (unit -> Tensor.t * Tensor.t) -> (unit -> Tensor.t) ->
    float -> int -> float -> unit
end

module Generator : Network
module Discriminator : Network
module WeightNetwork : Network
module FeatureNetwork : Network
module Classifier : Network
module Optimizer : Optimizer