open Torch
open Data_loader
open Utils
open Model

let wasserstein_distance mu nu p =
  let cost_matrix = Utils.pairwise_distances mu nu in
  let transport_plan = discrete_optimal_transport mu nu cost_matrix in
  Tensor.sum (Tensor.mul transport_plan (Tensor.pow cost_matrix (Tensor.of_float p)))
  |> Tensor.pow (Tensor.of_float (1. /. p))

let discrete_optimal_transport mu nu cost =
  let n, m = Tensor.shape2_exn cost in
  let a = Tensor.full [n] (1. /. float_of_int n) in
  let b = Tensor.full [m] (1. /. float_of_int m) in
  Utils.sinkhorn_knopp a b cost 0.01 100

let regularized_optimal_transport mu nu cost epsilon =
  let n, m = Tensor.shape2_exn cost in
  let a = Tensor.full [n] (1. /. float_of_int n) in
  let b = Tensor.full [m] (1. /. float_of_int m) in
  Utils.sinkhorn_knopp a b cost epsilon 100

let wasserstein_barycenter measures =
  let n = List.length measures in
  let d = Tensor.shape (List.hd measures) |> List.hd in
  let weights = Tensor.full [n] (1. /. float_of_int n) in
  let initial_barycenter = Tensor.mean (Tensor.stack measures 0) 0 in
  let max_iter = 100 in
  let rec iterate i barycenter =
    if i >= max_iter then barycenter
    else
      let transport_plans = List.map (fun measure -> 
        discrete_optimal_transport barycenter measure (Utils.pairwise_distances barycenter measure)
      ) measures in
      let new_barycenter = Tensor.mean (Tensor.stack (List.map2 Tensor.mm transport_plans measures) 0) 0 in
      iterate (i + 1) new_barycenter
  in
  iterate 0 initial_barycenter

let find_clean_representations dataset =
  let features, labels = dataset in
  let num_classes = Tensor.max labels |> Tensor.item |> int_of_float |> (+) 1 in
  let class_features = List.init num_classes (fun c ->
    Tensor.masked_select features (Tensor.eq labels (Tensor.of_int c))
  ) in
  wasserstein_barycenter class_features

let transport_noisy_labels clean_reps noisy_reps =
  let cost = Utils.pairwise_distances clean_reps noisy_reps in
  let transport_plan = regularized_optimal_transport clean_reps noisy_reps cost 0.1 in
  let _, predicted_labels = Tensor.max transport_plan 0 in
  predicted_labels

let filter dataset =
  let features, labels = dataset in
  let clean_reps = find_clean_representations dataset in
  let predicted_labels = transport_noisy_labels clean_reps features in
  features, predicted_labels

let train model dataset num_epochs batch_size learning_rate =
  let optimizer = Optimizer.adam (Model.parameters model) ~learning_rate in
  for epoch = 1 to num_epochs do
    let features, labels = dataset in
    let filtered_features, filtered_labels = filter dataset in
    let dataloader = DataLoader.create_dataloader (filtered_features, filtered_labels) ~batch_size in
    let total_loss = ref 0. in
    let num_batches = ref 0 in
    DataLoader.iter dataloader (fun batch ->
      let x, y = batch in
      let x_aug = Utils.augment_batch x in
      Optimizer.zero_grad optimizer;
      let pred = Model.forward model x_aug in
      let loss = Tensor.cross_entropy pred y in
      Tensor.backward loss;
      Optimizer.step optimizer;
      total_loss := !total_loss +. Tensor.float_value loss;
      num_batches := !num_batches + 1;
    );
    let avg_loss = !total_loss /. float_of_int !num_batches in
    if epoch mod 10 = 0 then
      Printf.printf "Epoch %d, Avg Loss: %f\n" epoch avg_loss
  done;
  model