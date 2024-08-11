open Base
open Torch

val compute_cost_matrix : Tensor.t -> Tensor.t -> (Tensor.t, string) Result.t

val optimal_transport :
  Tensor.t ->
  Tensor.t ->
  cost_matrix:Tensor.t ->
  eta_c:float ->
  eta_t:float ->
  labels:Tensor.t ->
  previous_mapping:Tensor.t option ->
  (Tensor.t, string) Result.t