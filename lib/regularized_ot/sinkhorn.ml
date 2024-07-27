open Torch

let algorithm source_dist target_dist cost_matrix lambda max_iter epsilon =
  let n_source = Tensor.shape source_dist |> List.hd in
  let n_target = Tensor.shape target_dist |> List.hd in
  
  if n_source <> (Tensor.shape cost_matrix |> List.hd) || 
     n_target <> (Tensor.shape cost_matrix |> List.tl |> List.hd) then
    failwith "Dimensions of distributions and cost matrix do not match";
  
  let k = Tensor.exp (Tensor.div cost_matrix (Tensor.f (-. lambda))) in
  
  let rec iterate u v iter =
    if iter >= max_iter then (u, v)
    else
      let u_new = Tensor.div source_dist (Tensor.mm k (Tensor.reshape v [n_target; 1])) in
      let v_new = Tensor.div target_dist (Tensor.mm (Tensor.transpose k ~dim0:0 ~dim1:1) (Tensor.reshape u_new [n_source; 1])) in
      
      if Tensor.max (Tensor.abs (Tensor.sub u_new u)) < epsilon &&
         Tensor.max (Tensor.abs (Tensor.sub v_new v)) < epsilon
      then (u_new, v_new)
      else iterate u_new v_new (iter + 1)
  in
  
  let u_init = Tensor.ones [n_source] in
  let v_init = Tensor.ones [n_target] in
  
  iterate u_init v_init 0