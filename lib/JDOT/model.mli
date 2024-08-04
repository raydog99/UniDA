open Torch

type t =
  | Linear of { input_dim : int; output_dim : int }
  | MLP of { input_dim : int; hidden_dims : int list; output_dim : int }

val create : t -> (Tensor.t -> Tensor.t) * Var_store.t
val to_string : t -> string
val of_json : Yojson.Safe.t -> t