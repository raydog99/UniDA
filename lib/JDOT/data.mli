open Torch

type t = {
  xs: Tensor.t;
  ys: Tensor.t;
  xt: Tensor.t;
  yt: Tensor.t option;
}

val load_data : source_file:string -> target_file:string -> t
val normalize_features : Tensor.t -> Tensor.t
val preprocess : t -> t
val create_batches : t -> int -> (Tensor.t * Tensor.t * Tensor.t) list