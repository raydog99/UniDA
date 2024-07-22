open Torch
open Grt
open Utils

type defining_function = Tensor.t -> Tensor.t -> Tensor.t

val gsw_distance : float -> Tensor.t -> Tensor.t -> defining_function -> Tensor.t -> Tensor.t

val max_gsw_distance : float -> Tensor.t -> Tensor.t -> defining_function -> Tensor.t ->
	num_iterations:int -> learning_rate:float -> Tensor.t

val project_gradient_descent : Tensor.t -> Tensor.t -> unit