open Base
open Torch
open Model

let gamma = 0.999
let tau = 0.95

let calculate_mitigating_vector (n_hat : Tensor.t) : Tensor.t =
  let b n = (1. -. gamma) /. (1. -. (gamma ** Float.(of_int (Tensor.to_int0_exn n)))) in
  Tensor.map1 (fun x -> Tensor.f (b x)) n_hat

let refine_soft_labels (soft_labels : Tensor.t) (mitigating_vector : Tensor.t) : Tensor.t =
  let refined = Tensor.mul soft_labels mitigating_vector in
  Tensor.div_scalar refined (Tensor.sum refined ~dim:[1] ~keepdim:true)

let estimate_class_distribution (predictions : Tensor.t) : Tensor.t =
  let above_threshold = Tensor.gt predictions (Tensor.f tau) in
  Tensor.sum above_threshold ~dim:[0] ~keepdim:false

let train (model : Model.t) (optimizer : Optimizer.t) 
          (labeled_data : (Tensor.t * Tensor.t) list) 
          (unlabeled_data : Tensor.t list) 
          (num_classes : int) (max_epochs : int) =
  let rec train_epoch epoch =
    if epoch >= max_epochs then model
    else begin
      List.iter labeled_data ~f:(fun (x, y) ->
        let loss = Model.supervised_loss model x y in
        Optimizer.backward_step optimizer ~loss);

      let pseudo_labels = List.map unlabeled_data ~f:(fun x ->
        let predictions = Model.predict model x in
        let n_hat = estimate_class_distribution predictions in
        let mitigating_vector = calculate_mitigating_vector n_hat in
        refine_soft_labels predictions mitigating_vector
      ) in

      List.iter2_exn unlabeled_data pseudo_labels ~f:(fun x y_pseudo ->
        let loss = Model.unsupervised_loss model x y_pseudo in
        Optimizer.backward_step optimizer ~loss);

      let accuracy = Utils.evaluate model labeled_data in
      Stdio.printf "Epoch %d: Accuracy = %.4f\n" epoch accuracy;

      train_epoch (epoch + 1)
    end
  in
  train_epoch 0