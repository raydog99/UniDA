open Torch
open Data
open Utils
open Optimal_transport

type t = {
  feature_extractor: Nn.t;
  classifier: Nn.t;
  alpha: float;
  lambda_t: float;
}

let create feature_extractor classifier alpha lambda_t =
  { feature_extractor; classifier; alpha; lambda_t }

let forward model x =
  let features = Nn.forward model.feature_extractor x in
  Nn.forward model.classifier features

let compute_cost model x_s y_s x_t =
  let z_s = Nn.forward model.feature_extractor x_s in
  let z_t = Nn.forward model.feature_extractor x_t in
  let y_t_pred = Nn.forward model.classifier z_t in
  
  let feature_cost = Tensor.mse_loss z_s z_t Reduction.None in
  let classification_cost = Tensor.cross_entropy y_s y_t_pred Reduction.None in
  
  Tensor.(model.alpha * feature_cost + model.lambda_t * classification_cost)

let train model source_data target_data num_epochs batch_size learning_rate =
  let optimizer = Optimizer.adam (Nn.parameters model.feature_extractor @ Nn.parameters model.classifier) ~lr:learning_rate in
  
  let source_dataloader = Data.create_dataloader source_data batch_size true in
  let target_dataloader = Data.create_dataloader target_data batch_size true in
  
  for epoch = 1 to num_epochs do
    let source_iter = ref 0 in
    let target_iter = ref 0 in
    let total_loss = ref 0.0 in
    
    while !source_iter < fst source_dataloader && !target_iter < fst target_dataloader do
      let x_s, y_s = snd source_dataloader !source_iter in
      let x_t, _ = snd target_dataloader !target_iter in
      
      Optimizer.zero_grad optimizer;
      
      let cost_matrix = compute_cost model x_s y_s x_t in
      let ot_matrix = compute_optimal_transport cost_matrix in
      
      let z_s = Nn.forward model.feature_extractor x_s in
      let z_t = Nn.forward model.feature_extractor x_t in
      
      let feature_loss = Tensor.sum Tensor.(ot_matrix * Tensor.mse_loss z_s z_t Reduction.None) in
      let classification_loss = Tensor.sum Tensor.(ot_matrix * Tensor.cross_entropy y_s (Nn.forward model.classifier z_t) Reduction.None) in
      let source_loss = Tensor.cross_entropy y_s (forward model x_s) Reduction.Mean in
      
      let loss = Tensor.(feature_loss + classification_loss + source_loss) in
      
      Tensor.backward loss;
      Optimizer.step optimizer;
      
      total_loss := !total_loss +. Tensor.float_value loss;
      
      source_iter := !source_iter + 1;
      target_iter := !target_iter + 1;
    done;
    
    Printf.printf "Epoch %d/%d, Loss: %.4f\n" epoch num_epochs (!total_loss /. float_of_int (fst source_dataloader));
  done

let adapt source_data target_data num_epochs batch_size learning_rate =
  let feature_extractor = Nn.sequential
    [ Nn.conv2d ~in_channels:1 ~out_channels:32 ~kernel_size:3 ~stride:1 ~padding:1 ()
    ; Nn.relu
    ; Nn.max_pool2d ~kernel_size:2
    ; Nn.conv2d ~in_channels:32 ~out_channels:64 ~kernel_size:3 ~stride:1 ~padding:1 ()
    ; Nn.relu
    ; Nn.max_pool2d ~kernel_size:2
    ; Nn.flatten
    ; Nn.linear ~in_features:3136 ~out_features:128 ()
    ] in
  
  let classifier = Nn.sequential
    [ Nn.linear ~in_features:128 ~out_features:10 ()
    ; Nn.log_softmax ~dim:1
    ] in
  
  let model = create feature_extractor classifier 0.1 1.0 in
  train model source_data target_data num_epochs batch_size learning_rate;
  model

let evaluate model dataset =
  let dataloader = Data.create_dataloader dataset 100 false in
  let num_correct = ref 0 in
  let num_total = ref 0 in
  
  for i = 0 to (fst dataloader) - 1 do
    let x, y = snd dataloader i in
    let y_pred = forward model x in
    let _, predicted = Tensor.max y_pred ~dim:1 ~keepdim:false in
    num_correct := !num_correct + Tensor.sum (Tensor.eq predicted y) |> Tensor.int_value;
    num_total := !num_total + Tensor.shape y |> List.hd;
  done;
  
  float_of_int !num_correct /. float_of_int !num_total