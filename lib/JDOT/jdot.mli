open Model
open Torch
open Data
open Config

type t

val create : Model.t -> Loss.t -> Config.t -> t
val compute_cost : t -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t -> float
val train_step : t -> Tensor.t -> Tensor.t -> Tensor.t -> float
val evaluate : t -> Data.t -> float
val fit : t -> Data.t -> unit
val predict : t -> Tensor.t -> Tensor.t
val save : t -> string -> unit
val load : string -> t