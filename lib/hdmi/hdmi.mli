open Torch

type t

val create : int -> int -> int -> (t, string) result
val train_source : t -> (Tensor.t * Tensor.t) list -> int -> float -> (unit, string) result
val adapt_target : t -> Tensor.t list -> int -> float -> (unit, string) result
val optimize : t -> Tensor.t list -> int -> float -> (unit, string) result