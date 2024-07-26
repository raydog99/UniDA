open Torch
open Data


type t

val create : Nn.t -> Nn.t -> float -> float -> t

val forward : t -> Tensor.t -> Tensor.t

val adapt : Data.dataset -> Data.dataset -> int -> int -> float -> t

val evaluate : t -> Data.dataset -> float