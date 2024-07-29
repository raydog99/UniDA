open Torch
open Utils
open Rejection
open Label_shift

type osda_params = {
  epsilon: float;
  rejection_threshold: float;
}

let osda x_s y_s x_t params =
  let cost = compute_cost x_s x_t in
  let mu_s = normalize_tensor (Tensor.ones [Tensor.size x_s 0]) in
  let mu_t = normalize_tensor (Tensor.ones [Tensor.size x_t 0]) in

  (* Rejection step *)
  let rejection_result = rejection ~epsilon:params.epsilon cost mu_s mu_t params.rejection_threshold in
  let x_t_filtered = apply_rejection rejection_result x_t in
  let accepted_indices = get_accepted_indices rejection_result in

  (* Label shift step *)
  let d = Tensor.one_hot y_s ~num_classes:(Tensor.size y_s 0) in
  let label_shift_result = label_shift ~epsilon:params.epsilon cost mu_s rejection_result.mu_t d in

  (* Predict target labels *)
  let target_labels = label_shift_result.predict_target_labels () in
  let class_proportions = get_class_proportions label_shift_result in

  (rejection_result, label_shift_result, target_labels, accepted_indices, class_proportions)

let predict x_s y_s x_t params =
  let (_, _, target_labels, accepted_indices, _) = osda x_s y_s x_t params in
  (target_labels, accepted_indices)