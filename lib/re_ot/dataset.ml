open Torch

type t = {
  features: Tensor.t;
  labels: Tensor.t;
  num_classes: int;
}

let create features labels num_classes =
  { features; labels; num_classes }

let batch t batch_size =
  let num_samples = Tensor.shape t.features |> Array.to_list |> List.hd in
  let indices = Tensor.randperm num_samples in
  let batch_indices = Tensor.narrow indices ~dim:0 ~start:0 ~length:batch_size in
  let batch_features = Tensor.index_select t.features ~dim:0 ~index:batch_indices in
  let batch_labels = Tensor.index_select t.labels ~dim:0 ~index:batch_indices in
  (batch_features, batch_labels)

let get_all t =
  (t.features, t.labels)