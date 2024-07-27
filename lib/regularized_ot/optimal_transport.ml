open Torch

open Torch

let compute_transport_plan source_features target_features lambda max_iter epsilon =
  Utils.check_input_dimensions source_features target_features;
  
  let cost_matrix = Tensor.cdist ~p:2 source_features target_features in
  let source_dist = Tensor.ones [Tensor.shape source_features |> List.hd] in
  let target_dist = Tensor.ones [Tensor.shape target_features |> List.hd] in
  
  let u, v = Sinkhorn.algorithm source_dist target_dist cost_matrix lambda max_iter epsilon in
  
  let transport_plan = Tensor.mul (Tensor.mul (Tensor.reshape u [-1; 1]) (Tensor.reshape v [1; -1])) 
                        (Tensor.exp (Tensor.div cost_matrix (Tensor.f (-. lambda)))) in
  
  transport_plan

let compute_transport_plan_default source_features target_features =
  compute_transport_plan source_features target_features 0.1 100 1e-6