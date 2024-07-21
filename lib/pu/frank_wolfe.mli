open Torch

type optimization_problem = Tensor.t -> float * Tensor.t

val optimize : 
    optimization_problem -> 
    Tensor.t -> 
    int -> 
    float -> 
    (int -> Tensor.t -> unit) option ->
    Tensor.t

val partial_gw_step : 
    PartialOT.cost_matrix -> 
    PartialOT.cost_matrix -> 
    Tensor.t -> 
    Tensor.t -> 
    float ->
    optimization_problem