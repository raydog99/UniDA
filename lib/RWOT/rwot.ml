open Torch

type t = {
  cnn_source: nn;
  cnn_target: nn;
  optimizer: Optimizer.t;
  mutable class_centers: Tensor.t;
}

let create input_shape num_classes learning_rate =
  let cnn_source = 
    Layer.sequential 
      [Layer.conv2d ~in_channels:3 ~out_channels:64 ~kernel_size:3 ~stride:1 ~padding:1 ();
       Layer.relu;
       Layer.max_pool2d ~kernel_size:2 ~stride:2 ();
       Layer.conv2d ~in_channels:64 ~out_channels:128 ~kernel_size:3 ~stride:1 ~padding:1 ();
       Layer.relu;
       Layer.max_pool2d ~kernel_size:2 ~stride:2 ();
       Layer.flatten ();
       Layer.linear (128 * (input_shape / 4) * (input_shape / 4)) 256;
       Layer.relu;
       Layer.linear 256 num_classes]
  in
  let cnn_target = Layer.copy cnn_source in
  let vs = Var_store.create ~name:"rwot" () in
  let optimizer = Optimizer.adam (Var_store.all_vars vs) ~learning_rate in
  let class_centers = Tensor.zeros [num_classes; 256] in
  { cnn_source; cnn_target; optimizer; class_centers }

let calculate_class_centers t source_samples source_labels =
  let features = Layer.apply t.cnn_source source_samples in
  let num_classes = Tensor.shape t.class_centers |> List.hd in
  t.class_centers <- Tensor.zeros [num_classes; 256];
  for c = 0 to num_classes - 1 do
    let class_mask = Tensor.(source_labels = scalar_i c) in
    let class_features = Tensor.masked_select features class_mask in
    let class_center = Tensor.mean class_features ~dim:[0] ~keepdim:true in
    t.class_centers <- Tensor.index_put t.class_centers [Some (Tensor.of_int0 c)] class_center
  done

let calculate_spatial_prototypical_matrix source_features target_features =
  let dist_matrix = Tensor.cdist source_features target_features in
  Tensor.neg dist_matrix

let sharpen_probability_annotation_matrix prob_matrix temperature =
  let exp_matrix = Tensor.exp (Tensor.div_scalar prob_matrix (Scalar.float temperature)) in
  Tensor.div exp_matrix (Tensor.sum exp_matrix ~dim:[1] ~keepdim:true)

let update_shrinking_subspace_reliability_cost_matrix q_matrix alpha =
  let row_sum = Tensor.sum q_matrix ~dim:[1] ~keepdim:true in
  let col_sum = Tensor.sum q_matrix ~dim:[0] ~keepdim:true in
  Tensor.mul q_matrix (Tensor.add (Tensor.div_scalar row_sum (Scalar.float alpha)) 
                                  (Tensor.div_scalar col_sum (Scalar.float alpha)))

let calculate_losses t source_samples source_labels target_samples =
  let source_features = Layer.apply t.cnn_source source_samples in
  let target_features = Layer.apply t.cnn_target target_samples in
  let d_matrix = calculate_spatial_prototypical_matrix source_features target_features in
  let m_matrix = sharpen_probability_annotation_matrix d_matrix 0.1 in
  let q_matrix = update_shrinking_subspace_reliability_cost_matrix m_matrix 0.1 in
  let l_g = Tensor.mean (Tensor.mul d_matrix q_matrix) in
  let l_p = Tensor.neg (Tensor.mean (Tensor.mul m_matrix (Tensor.log m_matrix))) in
  let source_logits = Layer.apply t.cnn_source source_samples in
  let l_cls = Tensor.cross_entropy source_logits source_labels in
  (l_g, l_p, l_cls)

let train t source_data target_data num_iterations batch_size nb =
  for i = 1 to num_iterations do
    let source_batch, source_labels = Dataset.random_batch source_data batch_size in
    let target_batch, _ = Dataset.random_batch target_data batch_size in
    
    calculate_class_centers t (Dataset.random_batch source_data nb |> fst) source_labels;
    
    let l_g, l_p, l_cls = calculate_losses t source_batch source_labels target_batch in
    let total_loss = Tensor.add (Tensor.add l_g l_p) l_cls in
    
    Optimizer.zero_grad t.optimizer;
    Tensor.backward total_loss;
    Optimizer.step t.optimizer;
    
    if i mod 100 = 0 then
      Printf.printf "Iteration %d: L_g = %f, L_p = %f, L_cls = %f\n" 
        i (Tensor.to_float0_exn l_g) (Tensor.to_float0_exn l_p) (Tensor.to_float0_exn l_cls)
  done