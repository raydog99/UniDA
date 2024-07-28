open Torch

type t

val create_preact_resnet18 : int -> t
val forward : t -> Tensor.t -> Tensor.t
val parameters : t -> Tensor.t list