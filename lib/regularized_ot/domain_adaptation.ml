open Torch

let adapt source_features target_features lambda max_iter epsilon =
  Utils.check_input_dimensions source_features target_features;
  let normalized_source = Utils.normalize_features source_features in
  let normalized_target = Utils.normalize_features target_features in
  let transport_plan = Optimal_transport.compute_transport_plan normalized_source normalized_target lambda max_iter epsilon in
  let adapted_features = Tensor.mm transport_plan normalized_target in
  adapted_features

let adapt_default source_features target_features =
  adapt source_features target_features 0.1 100 1e-6

let adapt_with_transport_plan source_features target_features lambda max_iter epsilon =
  Utils.check_input_dimensions source_features target_features;
  let normalized_source = Utils.normalize_features source_features in
  let normalized_target = Utils.normalize_features target_features in
  let transport_plan = Optimal_transport.compute_transport_plan normalized_source normalized_target lambda max_iter epsilon in
  let adapted_features = Tensor.mm transport_plan normalized_target in
  (adapted_features, transport_plan)