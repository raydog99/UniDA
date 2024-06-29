open Torch

module Generator = struct
  let model = 
    let open NN in
    sequential [
      linear ~in_features:784 ~out_features:256 ();
      relu ();
      linear ~in_features:256 ~out_features:128 ();
      relu ();
    ]
  
  let forward x = NN.apply model x
end

module Classifier = struct
  let model =
    let open NN in
    sequential [
      linear ~in_features:128 ~out_features:64 ();
      relu ();
      linear ~in_features:64 ~out_features:10 ();
      softmax ~dim:(-1) ();
    ]
  
  let forward x = NN.apply model x
end

module KantorovichPotential = struct
  let model = 
    let open NN in
    sequential [
      linear ~in_features:128 ~out_features:64 ();
      relu ();
      linear ~in_features:64 ~out_features:1 ();
    ]
  
  let forward x = NN.apply model x
end

let cross_entropy_loss predictions targets =
  Tensor.((-targets * log predictions) |> sum)

let wasserstein_distance x_source x_target h_source h_target phi lambda epsilon =
  let z_source = Generator.forward x_source in
  let z_target = Generator.forward x_target in
  let pred_source = Classifier.forward z_source in
  let pred_target = Classifier.forward z_target in
  
  let cost_matrix = Tensor.(
    (cdist ~p:2. z_source z_target) * float lambda +
    (cdist ~p:2. pred_source pred_target)
  ) in
  
  let phi_source = KantorovichPotential.forward z_source in
  let phi_target = KantorovichPotential.forward z_target in
  
  let exp_term = Tensor.(
    (phi_target - cost_matrix) / float epsilon
    |> exp |> sum ~dim:[1] |> log
  ) in
  
  Tensor.(
    mean phi_source - mean (phi_target - float epsilon * exp_term)
  )

let train ~num_epochs ~batch_size ~learning_rate ~alpha ~lambda ~epsilon
    ~x_source ~y_source ~x_target =
  
  let optimizer = Optimizer.adam (Generator.model @ Classifier.model @ KantorovichPotential.model) ~learning_rate in
  
  for epoch = 1 to num_epochs do
    for batch = 0 to (Tensor.shape x_source).(0) / batch_size - 1 do
      let start_idx = batch * batch_size in
      let end_idx = (batch + 1) * batch_size in
      
      let x_source_batch = Tensor.narrow x_source ~dim:0 ~start:start_idx ~length:batch_size in
      let y_source_batch = Tensor.narrow y_source ~dim:0 ~start:start_idx ~length:batch_size in
      let x_target_batch = Tensor.narrow x_target ~dim:0 ~start:start_idx ~length:batch_size in
      
      let z_source = Generator.forward x_source_batch in
      let pred_source = Classifier.forward z_source in
      
      let loss_source = cross_entropy_loss pred_source y_source_batch in
      
      let ws_distance = wasserstein_distance x_source_batch x_target_batch 
                          (Classifier.forward) (Classifier.forward) 
                          KantorovichPotential.forward lambda epsilon in
      
      let total_loss = Tensor.(loss_source + float alpha * ws_distance) in
      
      Optimizer.zero_grad optimizer;
      Tensor.backward total_loss;
      Optimizer.step optimizer;
    done;
  done