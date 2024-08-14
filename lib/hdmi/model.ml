open Torch
open Feature_extractor
open Classifier

type t = {
  feature_extractor: Feature_extractor.t;
  classifiers: Classifier.t array;
}

let create input_channels num_classes num_hypotheses =
  match Feature_extractor.create input_channels with
  | Error e -> Error e
  | Ok feature_extractor ->
    try
      let classifiers = Array.init num_hypotheses (fun _ ->
        match Classifier.create 256 num_classes with
        | Ok c -> c
        | Error e -> raise (Failure e)
      ) in
      Ok { feature_extractor; classifiers }
    with
    | exn -> Error (Printf.sprintf "Failed to create model: %s" (Printexc.to_string exn))

let forward t x =
  match Feature_extractor.forward t.feature_extractor x with
  | Error e -> Error e
  | Ok features ->
    try
      let results = Array.map (fun classifier ->
        match Classifier.forward classifier features with
        | Ok result -> result
        | Error e -> raise (Failure e)
      ) t.classifiers in
      Ok results
    with
    | exn -> Error (Printf.sprintf "Forward pass failed in model: %s" (Printexc.to_string exn))

let parameters t =
  let fe_params = Layer.parameters t.feature_extractor in
  let classifier_params = Array.fold_left (fun acc c -> acc @ Layer.parameters c) [] t.classifiers in
  fe_params @ classifier_params