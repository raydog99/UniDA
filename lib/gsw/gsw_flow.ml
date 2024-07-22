open Torch
open Gsw

let flow target_dist initial_dist defining_func omega_theta ~num_iterations ~learning_rate ~p =
  let optimizer = Optimizer.adam [initial_dist] ~learning_rate in
  
  let rec optimize iter current_dist =
    if iter >= num_iterations then current_dist
    else
      let loss = GSW.gsw_distance p current_dist target_dist defining_func omega_theta in
      Optimizer.zero_grad optimizer;
      Tensor.backward loss;
      Optimizer.step optimizer;
      
      optimize (iter + 1) current_dist
  in
  
  optimize 0 initial_dist