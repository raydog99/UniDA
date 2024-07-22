open Torch
open Grt
open Utils

type defining_function = Tensor.t -> Tensor.t -> Tensor.t

let gsw_distance p mu nu defining_func omega_theta =
	let grt_mu = Grt.transform mu defining_func omega_theta in
	let grt_nu = Grt.transform nu defining_func omega_theta in

let distance_fn slice_mu slice_nu =
  let sorted_mu = Utils.sort_tensor slice_mu in
  let sorted_nu = Utils.sort_tensor slice_nu in
  Utils.l_p_distance p sorted_mu sorted_nu
in

Tensor.mean (Tensor.map2 distance_fn grt_mu grt_nu)

let max_gsw_distance p mu nu defining_func omega_theta ~num_iterations ~learning_rate =
	let optimizer = Optimizer.adam [omega_theta] ~learning_rate in

let rec optimize iter best_distance best_theta =
  if iter >= num_iterations then best_theta
  else
    let loss = gsw_distance p mu nu defining_func omega_theta in
    Optimizer.zero_grad optimizer;
    Tensor.backward loss;
    Optimizer.step optimizer;
    
    let new_best_distance, new_best_theta =
      if Tensor.to_float0_exn loss > Tensor.to_float0_exn best_distance
      then loss, omega_theta
      else best_distance, best_theta
    in
    
    optimize (iter + 1) new_best_distance new_best_theta
in

let initial_distance = gsw_distance p mu nu defining_func omega_theta in
let best_theta = optimize 0 initial_distance omega_theta in

gsw_distance p mu nu defining_func best_theta

let project_gradient_descent theta omega_theta =
	let normalized = Tensor.div theta (Tensor.norm theta ~dim:[0] ~p:2 ~keepdim:true) in
	Tensor.copy_ omega_theta normalized