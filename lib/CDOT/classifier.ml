open Base
open Torch

type t = {
  model: Module.t;
  optimizer: Optimizer.t;
}

let create input_dim num_classes =
  try
    let model = 
      Module.sequential
        [
          Module.linear ~input_dim ~output_dim:50;
          Module.relu ();
          Module.linear ~input_dim:50 ~output_dim:num_classes;
        ]
    in
    let optimizer = Optimizer.adam (Module.parameters model) ~learning_rate:0.01 in
    Ok { model; optimizer }
  with
  | _ -> Error "Failed to create classifier"

let train classifier data =
  try
    let { model; optimizer } = classifier in
    let num_epochs = 100 in
    for _ = 1 to num_epochs do
      Optimizer.zero_grad optimizer;
      let output = Module.forward model data.Data.features in
      let loss = Tensor.cross_entropy_for_logits output data.Data.labels in
      Tensor.backward loss;
      Optimizer.step optimizer
    done;
    Ok classifier
  with
  | _ -> Error "Failed to train classifier"

let evaluate classifier data =
  try
    let { model; _ } = classifier in
    let output = Module.forward model data.Data.features in
    let predicted = Tensor.argmax output ~dim:1 ~keepdim:false in
    let correct = Tensor.eq predicted data.Data.labels in
    let accuracy = Tensor.to_float0_exn (Tensor.mean correct |> Tensor.mul_scalar 100.0) in
    Ok accuracy
  with
  | _ -> Error "Failed to evaluate classifier"