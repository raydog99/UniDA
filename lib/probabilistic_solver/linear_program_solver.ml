open Torch

let solve cost_matrix r s =
  let m, n = Tensor.shape2_exn cost_matrix in
  
  let transport_plan = Tensor.zeros [m; n] in
  
  let rec fill_plan i j remaining_r remaining_s =
    if i >= m || j >= n then transport_plan
    else
      let amount = min (Tensor.get remaining_r [i]) (Tensor.get remaining_s [j]) in
      Tensor.set_ transport_plan [i; j] amount;
      let new_remaining_r = Tensor.set remaining_r [i] (Tensor.get remaining_r [i] -. amount) in
      let new_remaining_s = Tensor.set remaining_s [j] (Tensor.get remaining_s [j] -. amount) in
      if Tensor.get new_remaining_r [i] <= 1e-10 then
        fill_plan (i + 1) j new_remaining_r new_remaining_s
      else if Tensor.get new_remaining_s [j] <= 1e-10 then
        fill_plan i (j + 1) new_remaining_r new_remaining_s
      else
        fill_plan i j new_remaining_r new_remaining_s
  in
  
  let initial_plan = fill_plan 0 0 r s in
  
  let optimize plan =
    let iterations = 100 in
    let learning_rate = 0.01 in
    
    let rec optimize_step plan iter =
      if iter = 0 then plan
      else
        let gradient = Tensor.(plan * cost_matrix) in
        let new_plan = Tensor.(plan - (gradient * f learning_rate)) in
        let normalized_plan = project_transport_plan new_plan r s in
        optimize_step normalized_plan (iter - 1)
    in
    
    optimize_step plan iterations
  in
  
  optimize initial_plan

let project_transport_plan plan r s =
  let m, n = Tensor.shape2_exn plan in
  
  let row_sums = Tensor.sum plan ~dim:[1] ~keepdim:true in
  let row_scaled_plan = Tensor.(plan * (r / row_sums)) in
  
  let col_sums = Tensor.sum row_scaled_plan ~dim:[0] ~keepdim:true in
  Tensor.(row_scaled_plan * (s / col_sums))