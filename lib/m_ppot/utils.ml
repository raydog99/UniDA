open Torch
open Types
open Classifier

let update_alpha (old_alpha : float) (new_alpha : float) (lambda : float) : float =
  old_alpha *. (1. -. lambda) +. new_alpha *. lambda

let update_beta (old_beta : float) (new_beta : float) (lambda : float) : float =
  old_beta *. (1. -. lambda) +. new_beta *. lambda

let compute_alpha (logits : Tensor.t) (tau : float) : float =
  let confidence = Tensor.max logits ~dim:1 |> fst in
  let num_known = Tensor.(sum (confidence >= Scalar.f tau) |> to_float0_exn) in
  num_known /. float_of_int (Tensor.shape logits |> fst)

let compute_beta (weights : Tensor.t) (tau : float) : float =
  let num_common = Tensor.(sum (weights >= Scalar.f tau) |> to_float0_exn) in
  num_common /. float_of_int (Tensor.shape weights |> fst)

let create_training_state (config : config) : training_state =
  let feature_extractor = Feature_extractor.create () in
  let classifier = Classifier.create 2048 config.num_classes in
  let model = { feature_extractor; classifier } in
  let optimizer = Optimizer.adam (Nn.Module.parameters feature_extractor @ Nn.Module.parameters classifier) ~lr:config.learning_rate in
  { model; optimizer; alpha = config.alpha; beta = config.beta }