open Torch
open Utils
open Model
open Dataset
open Long_tailed

let compute_cost_matrix features labels =
  let distance = Utils.cosine_distance features labels in
  Tensor.sub (Tensor.ones (Tensor.shape distance)) distance

let relative_entropy_regularization p q =
  Utils.kl_divergence p q

let sinkhorn_algorithm cost_matrix a b q epsilon max_iter =
  let k = Tensor.exp (Tensor.div (Tensor.neg cost_matrix) epsilon) in
  let k = Tensor.mul k q in
  let u = Tensor.ones [| Tensor.shape a |> Array.to_list |> List.hd |] in
  let v = Tensor.ones [| Tensor.shape b |> Array.to_list |> List.hd |] in
  
  let rec iterate i u v =
    if i >= max_iter then (u, v)
    else
      let u_new = Tensor.div a (Tensor.matmul k v) in
      let v_new = Tensor.div b (Tensor.matmul (Tensor.transpose k ~dim0:0 ~dim1:1) u_new) in
      iterate (i + 1) u_new v_new
  in
  
  let u_final, v_final = iterate 0 u v in
  Tensor.mul (Tensor.mul (Tensor.diag u_final) k) (Tensor.diag v_final)

let re_ot_optimize cost_matrix a b q epsilon max_iter =
  sinkhorn_algorithm cost_matrix a b q epsilon max_iter

let inverse_re_ot features labels q epsilon max_iter =
  let cost_matrix = compute_cost_matrix features labels in
  let a = Tensor.ones [| Tensor.shape features |> Array.to_list |> List.hd |] in
  let b = Tensor.ones [| Tensor.shape labels |> Array.to_list |> List.hd |] in
  re_ot_optimize cost_matrix a b q epsilon max_iter

let re_ot_loss p_theta p_true =
  Utils.kl_divergence p_true p_theta

let train model dataset num_epochs learning_rate batch_size epsilon max_iter =
  let optimizer = Optimizer.adam (Model.parameters model) ~lr:learning_rate in
  
  for epoch = 1 to num_epochs do
    let features, labels = Dataset.batch dataset batch_size in
    let logits = Model.forward model features in
    let q = Utils.softmax logits 1 in
    
    let p_theta = inverse_re_ot features labels q epsilon max_iter in
    let loss = re_ot_loss p_theta labels in
    
    Optimizer.zero_grad optimizer;
    Tensor.backward loss;
    Optimizer.step optimizer;
    
    if epoch mod 10 = 0 then
      Printf.printf "Epoch %d, Loss: %f\n" epoch (Tensor.to_float0_exn loss)
  done

let infer model features =
  let logits = Model.forward model features in
  Utils.softmax logits 1

let update_q q uniform_q ratio_q epoch total_epochs =
  let t = float_of_int epoch /. float_of_int total_epochs in
  Tensor.add (Tensor.mul uniform_q (Tensor.float_vec [|1.0 -. t|])) (Tensor.mul ratio_q (Tensor.float_vec [|t|]))