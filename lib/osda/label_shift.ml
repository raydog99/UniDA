open Torch
open Utils

type label_shift_result = {
  pi: Tensor.t;
  nu: Tensor.t;
  predict_target_labels: unit -> Tensor.t;
}

let label_shift ?(epsilon=1e-3) cost mu_s mu_t d =
  if not (check_probability_vector mu_s && check_probability_vector mu_t) then
    raise (Invalid_argument "mu_s and mu_t must be probability vectors")
  else
    let pi = sinkhorn ~epsilon cost mu_s mu_t in
    let nu = Tensor.(matmul (transpose d 0 1) (sum pi 1)) in
    let nu_normalized = normalize_tensor nu in
    
    let predict_target_labels () =
      let class_distribution = Tensor.(matmul (transpose d 0 1) pi) in
      Tensor.argmax class_distribution ~dim:0 ~keepdim:false
    in
    
    { pi; nu = nu_normalized; predict_target_labels }

let get_class_proportions result =
  result.nu