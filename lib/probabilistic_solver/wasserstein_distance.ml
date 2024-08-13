open Torch
open Linear_program_solver

let compute_wasserstein_distance r s p =
  let cost_matrix = Tensor.(cdist ~p:2 r s) in
  let transport_plan = Linear_program_solver.solve cost_matrix r s in
  let transport_cost = Tensor.(sum (transport_plan * cost_matrix)) in
  Tensor.pow transport_cost (Scalar.f (1. /. p)) |> Tensor.to_float0_exn

let compute_pairwise_distances x y p =
  Tensor.cdist ~p x y

let compute_diameter x =
  let pairwise_distances = compute_pairwise_distances x x 2 in
  Tensor.max pairwise_distances |> Tensor.to_float0_exn

let euclidean_distance x y =
  Tensor.(sqrt (sum (pow (x - y) (Scalar.i 2))))