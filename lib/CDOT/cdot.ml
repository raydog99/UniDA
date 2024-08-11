open Base
open Torch
open Data
open Ot
open Classifier

let adapt source_data target_data eta_c eta_t previous_mapping =
  let open Result.Let_syntax in
  let%bind cost_matrix = Ot.compute_cost_matrix source_data.Data.features target_data.Data.features in
  let%bind coupling = Ot.optimal_transport 
    source_data.Data.features 
    target_data.Data.features 
    ~cost_matrix 
    ~eta_c 
    ~eta_t 
    ~labels:source_data.Data.labels
    ~previous_mapping
  in
  try
    let mapped_source = Tensor.matmul coupling target_data.Data.features in
    Ok { Data.features = mapped_source; labels = source_data.Data.labels }
  with
  | _ -> Error "Failed to compute mapped source data"

let train_and_evaluate classifier source_data target_data =
  let open Result.Let_syntax in
  let%bind trained_classifier = Classifier.train classifier source_data in
  Classifier.evaluate trained_classifier target_data

let run source_data target_data_seq eta_c eta_t =
  let rec loop acc current_source previous_mapping = function
    | [] -> Ok (List.rev acc)
    | target :: rest ->
        let open Result.Let_syntax in
        let%bind adapted_source = adapt current_source target eta_c eta_t previous_mapping in
        let%bind accuracy = train_and_evaluate (Classifier.create 2 2) adapted_source target in
        let%bind new_previous_mapping = Ot.compute_cost_matrix current_source.Data.features adapted_source.Data.features in
        loop (accuracy :: acc) adapted_source (Some new_previous_mapping) rest
  in
  loop [] source_data None target_data_seq