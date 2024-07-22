open Torch

type t =
  | NegativeEntropy of float
  | SquaredNorm of float
  | GroupLasso of float * float * (int list) list

val apply : t -> Tensor.t -> Tensor.t
val validate : t -> unit