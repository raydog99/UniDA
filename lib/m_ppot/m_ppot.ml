open Torch
open Types
open Classifier
open Prototypes
open Optimal_transport
open Losses

let train_batch (state : training_state) (config : config) 
                (source_batch : Tensor.t) (source_labels_batch : Tensor.t) (target_batch : Tensor.t) : loss =
  let source_features = Feature_extractor.extract_features state.model.feature_extractor source_batch in
  let target_features = Feature_extractor.extract_features state.model.feature_extractor target_batch in
  
  let source_logits = Classifier.classify state.model.classifier source_features in
  let target_logits = Classifier.classify state.model.classifier target_features in

  let prototypes = Prototypes.compute_prototypes source_features source_labels_batch config.num_classes in

  let m_ppot = Losses.m_ppot_loss prototypes target_features state.alpha in
  
  let transport_plan = Optimal_transport.solve_pot (Optimal_transport.compute_cost_matrix prototypes target_features) state.alpha in
  let target_weights = Tensor.sum transport_plan ~dim:[0] |> fun x -> Tensor.mul x (Scalar.f (float_of_int config.num_classes /. state.alpha)) in
  let source_weights = Tensor.sum transport_plan ~dim:[1] |> fun x -> Tensor.mul x (Scalar.f (float_of_int (Tensor.shape target_features |> fst) /. state.alpha)) in

  let entropy = Losses.reweighted_entropy_loss target_logits target_weights in
  let cross_entropy = Losses.reweighted_cross_entropy_loss source_logits source_labels_batch source_weights in

  Losses.compute_loss m_ppot entropy cross_entropy config

let train_epoch (state : training_state) (config : config) 
                (source_data : Tensor.t) (source_labels : Tensor.t) (target_data : Tensor.t) : unit =
  let num_batches = (Tensor.shape source_data |> fst) / config.batch_size in

  for batch = 1 to num_batches do
    let source_batch = Tensor.narrow source_data ~dim:0 ~start:((batch - 1) * config.batch_size) ~length:config.batch_size in
    let source_labels_batch = Tensor.narrow source_labels ~dim:0 ~start:((batch - 1) * config.batch_size) ~length:config.batch_size in
    let target_batch = Tensor.narrow target_data ~dim:0 ~start:((batch - 1) * config.batch_size) ~length:config.batch_size in

    let loss = train_batch state config source_batch source_labels_batch target_batch in

    Optimizer.zero_grad state.optimizer;
    Tensor.backward loss.total;
    Optimizer.step state.optimizer;

    let new_alpha = Utils.compute_alpha target_logits config.tau_1 in
    let new_beta = Utils.compute_beta source_weights config.tau_2 in
    state.alpha <- Utils.update_alpha state.alpha new_alpha 0.001;
    state.beta <- Utils.update_beta state.beta new_beta 0.001;
  done

let train (config : config) (source_data : Tensor.t) (source_labels : Tensor.t) (target_data : Tensor.t) : unit =
  let state = Utils.create_training_state config in

  for epoch = 1 to config.num_epochs do
    train_epoch state config source_data source_labels target_data;

    Printf.printf "Epoch %d, Alpha: %f, Beta: %f\n" epoch state.alpha state.beta;
  done