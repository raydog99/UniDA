open Torch
open Model

type t = {
  source_model: Model.t;
  target_model: Model.t;
  num_hypotheses: int;
}

let create input_channels num_classes num_hypotheses =
  match Model.create input_channels num_classes num_hypotheses with
  | Error e -> Error e
  | Ok model ->
    Ok { source_model = model; target_model = model; num_hypotheses }

let train_source t source_data epochs learning_rate =
  let loss_fn = Layer.cross_entropy_loss () in
  let optimizer = Optimizer.adam (Model.parameters t.source_model) ~lr:learning_rate in
  
  try
    for epoch = 1 to epochs do
      List.iter (fun (x, y) ->
        match Model.forward t.source_model x with
        | Error e -> raise (Failure e)
        | Ok y_pred ->
          let y_pred = Array.map (fun pred -> Tensor.squeeze pred ~dim:[0]) y_pred in
          let loss = Array.fold_left (fun acc pred -> acc +. Layer.cross_entropy_loss pred y) 0. y_pred in
          Optimizer.backward optimizer (Tensor.of_float0 loss);
          Optimizer.step optimizer;
          Optimizer.zero_grad optimizer
      ) source_data;
      if epoch mod 10 = 0 then
        Printf.printf "Epoch %d completed\n" epoch
    done;
    Ok ()
  with
  | exn -> Error (Printf.sprintf "Source training failed: %s" (Printexc.to_string exn))

let compute_mutual_information t x =
  match Model.forward t.target_model x with
  | Error e -> Error e
  | Ok predictions ->
    try
      let batch_size = Tensor.shape x |> List.hd in
      let num_classes = Tensor.shape (Array.get predictions 0) |> List.last in
      
      let entropy_y = 
        Array.fold_left (fun acc pred ->
          let mean_pred = Tensor.mean pred ~dim:[0] ~keepdim:true in
          let log_mean_pred = Tensor.log mean_pred in
          acc -. (Tensor.sum (Tensor.mul mean_pred log_mean_pred) |> Tensor.to_float0_exn)
        ) 0. predictions in
      
      let entropy_y_given_x =
        Array.fold_left (fun acc pred ->
          let log_pred = Tensor.log pred in
          acc -. (Tensor.sum (Tensor.mul pred log_pred) |> Tensor.to_float0_exn) /. float_of_int batch_size
        ) 0. predictions in
      
      Ok (entropy_y -. entropy_y_given_x)
    with
    | exn -> Error (Printf.sprintf "Mutual information computation failed: %s" (Printexc.to_string exn))

let compute_hypothesis_disparity t x =
  match Model.forward t.target_model x with
  | Error e -> Error e
  | Ok predictions ->
    try
      let anchor_idx = Random.int t.num_hypotheses in
      let anchor_pred = Array.get predictions anchor_idx in
      
      let total = Array.fold_left (fun acc idx ->
        if idx = anchor_idx then acc
        else
          let pred = Array.get predictions idx in
          let ce_loss = Layer.cross_entropy_loss pred anchor_pred in
          acc +. (Tensor.to_float0_exn ce_loss)
      ) 0. (Array.init t.num_hypotheses (fun i -> i)) in
      
      Ok (total /. float_of_int (t.num_hypotheses - 1))
    with
    | exn -> Error (Printf.sprintf "Hypothesis disparity computation failed: %s" (Printexc.to_string exn))

let adapt_target t target_data epochs learning_rate =
  let optimizer = Optimizer.adam (Model.parameters t.target_model) ~lr:learning_rate in
  
  try
    for epoch = 1 to epochs do
      List.iter (fun x ->
        match compute_mutual_information t x, compute_hypothesis_disparity t x with
        | Ok mi, Ok hd ->
          let loss = Tensor.neg (Tensor.of_float0 mi) +. Tensor.of_float0 hd in
          Optimizer.backward optimizer loss;
          Optimizer.step optimizer;
          Optimizer.zero_grad optimizer
        | Error e, _ | _, Error e -> raise (Failure e)
      ) target_data;
      if epoch mod 10 = 0 then
        Printf.printf "Adaptation epoch %d completed\n" epoch
    done;
    Ok ()
  with
  | exn -> Error (Printf.sprintf "Target adaptation failed: %s" (Printexc.to_string exn))

let optimize = adapt_target