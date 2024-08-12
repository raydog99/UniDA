open Torch

type t = Optimizer.t

val adam : ?beta1:float -> ?beta2:float -> ?eps:float -> ?weight_decay:float -> ?amsgrad:bool -> lr:float -> Layer.t list -> t
val sgd : ?momentum:float -> ?dampening:float -> ?weight_decay:float -> ?nesterov:bool -> lr:float -> Layer.t list -> t
val zero_grad : t -> unit
val step : t -> unit