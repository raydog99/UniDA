open Base
open Torch

let balanced_accuracy predictions labels num_classes =
  let confusion_matrix = Tensor.zeros [num_classes; num_classes] in
  Tensor.iteri predictions ~f:(fun i pred ->
    let true_label = Tensor.get_int1 labels i in
    let pred_label = Tensor.argmax pred ~dim:0 ~keepdim:false |> Tensor.to_int0_exn in
    let current = Tensor.get confusion_matrix [true_label; pred_label] in
    Tensor.set confusion_matrix [true_label; pred_label] (current +. 1.)
  );
  
  let class_accuracies = List.init num_classes ~f:(fun i ->
    let row = Tensor.slice confusion_matrix ~dim:0 ~start:i ~end_:(i+1) in
    let total = Tensor.sum row in
    if Float.(total > 0.) then Tensor.get row [0; i] /. total else 0.
  ) in
  
  List.fold class_accuracies ~init:0. ~f:(+.) /. Float.of_int num_classes

let evaluate model data =
  let predictions, labels = 
    List.fold data ~init:([], []) ~f:(fun (preds, labs) (x, y) ->
      let pred = Model.predict model x in
      (pred :: preds, y :: labs)
    )
  in
  let predictions = Tensor.cat (List.rev predictions) ~dim:0 in
  let labels = Tensor.cat (List.rev labels) ~dim:0 in
  balanced_accuracy predictions labels 10  