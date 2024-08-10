open Torch

type t
type config = {
  latent_dim : int;
  data_dim : int;
  learning_rate_g : float;
  learning_rate_d : float;
  beta1 : float;
  beta2 : float;
}

val create : config -> Device.t -> t
val to_device : t -> t
val generate : t -> int -> Tensor.t
val discriminate : t -> Tensor.t -> Tensor.t * Tensor.t
val train_step : t -> Tensor.t -> Optimizer.t -> Optimizer.t -> float * float
val evaluate : t -> (Tensor.t, Tensor.t) Data.batch_stream -> float
val save : t -> string -> unit
val load : config -> Device.t -> string -> t