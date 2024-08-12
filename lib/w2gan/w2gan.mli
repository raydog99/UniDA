open Torch
open Data

type t

val create : int -> int list -> int -> float -> float -> t
val forward_map : t -> Tensor.t -> Tensor.t
val inverse_map : t -> Tensor.t -> Tensor.t
val total_loss : t -> Tensor.t -> Tensor.t -> Tensor.t
val train : t -> num_epochs:int -> batch_size:int -> data_loader:Dataset.t -> validation_loader:Dataset.t -> t
val sample : t -> int -> Tensor.t
val save : t -> string -> unit
val load : string -> float -> t