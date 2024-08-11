open Base
open Torch

let compute_cost_matrix x_source x_target =
  try
    let x_source_squared = Tensor.sum (Tensor.pow x_source 2.0) ~dim:[1] ~keepdim:true in
    let x_target_squared = Tensor.sum (Tensor.pow x_target 2.0) ~dim:[1] ~keepdim:true in
    let product = Tensor.matmul x_source x_target ~transpose_b:true in
    Ok (Tensor.sub (Tensor.add x_source_squared (Tensor.transpose x_target_squared ~dim0:0 ~dim1:1)) (Tensor.mul_scalar product 2.0))
  with
  | _ -> Error "Failed to compute cost matrix"

let sinkhorn coupling cost_matrix mu nu epsilon max_iter =
  let rec loop iter coupling =
    if iter >= max_iter then Ok coupling
    else
      try
        let u = Tensor.div mu (Tensor.sum coupling ~dim:[1] ~keepdim:true) in
        let coupling = Tensor.mul coupling (Tensor.exp (Tensor.div (Tensor.log u) epsilon)) in
        let v = Tensor.div nu (Tensor.sum coupling ~dim:[0] ~keepdim:true) in
        let coupling = Tensor.mul coupling (Tensor.exp (Tensor.div (Tensor.log v) epsilon)) in
        loop (iter + 1) coupling
      with
      | _ -> Error "Sinkhorn algorithm failed"
  in
  loop 0 coupling

let group_lasso_regularizer coupling labels =
  try
    let unique_labels = Tensor.unique labels in
    let n_classes = Tensor.shape unique_labels |> List.hd_exn in
    let regularization = Tensor.zeros [] in
    for i = 0 to n_classes - 1 do
      let class_mask = Tensor.eq labels (Tensor.get unique_labels [|i|]) in
      let class_coupling = Tensor.masked_select coupling class_mask in
      let group_norm = Tensor.norm class_coupling ~p:2 ~dim:[0] in
      Tensor.(regularization += sum group_norm)
    done;
    Ok regularization
  with
  | _ -> Error "Failed to compute group lasso regularizer"

let time_regularizer coupling previous_mapping =
  match previous_mapping with
  | None -> Ok (Tensor.zeros [])
  | Some prev_map ->
      try
        let diff = Tensor.sub coupling prev_map in
        Ok (Tensor.sum (Tensor.pow diff 2.0))
      with
      | _ -> Error "Failed to compute time regularizer"

let optimal_transport x_source x_target ~cost_matrix ~eta_c ~eta_t ~labels ~previous_mapping =
  let open Result.Let_syntax in
  let n_source, n_target = Tensor.shape x_source |> List.hd_exn, Tensor.shape x_target |> List.hd_exn in
  let mu = Tensor.ones [n_source] |> Tensor.div_scalar (Float.of_int n_source) in
  let nu = Tensor.ones [n_target] |> Tensor.div_scalar (Float.of_int n_target) in
  let initial_coupling = Tensor.mul (Tensor.outer mu nu) (Float.of_int (n_source * n_target)) in
  let epsilon = 0.1 in
  let max_iter = 100 in

  let rec optimize iter coupling =
    if iter >= max_iter then Ok coupling
    else
      let%bind coupling = sinkhorn coupling cost_matrix mu nu epsilon 10 in
      let%bind class_reg = group_lasso_regularizer coupling labels in
      let%bind time_reg = time_regularizer coupling previous_mapping in
      try
        let total_cost = Tensor.sum (Tensor.mul coupling cost_matrix) +
                         Tensor.mul_scalar class_reg eta_c +
                         Tensor.mul_scalar time_reg eta_t in
        let grad = Tensor.grad total_cost [|coupling|] in
        let coupling = Tensor.sub coupling (Tensor.mul_scalar (Tensor.get grad 0) 0.01) in
        optimize (iter + 1) coupling
      with
      | _ -> Error "Failed to optimize coupling"
  in
  optimize 0 initial_coupling