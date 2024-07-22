open Torch

val flow : Tensor.t -> Tensor.t -> (Tensor.t -> Tensor.t -> Tensor.t) -> Tensor.t ->
	num_iterations:int -> learning_rate:float -> p:float -> Tensor.t