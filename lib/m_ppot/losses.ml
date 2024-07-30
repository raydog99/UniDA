open Torch
open Types
open Optimal_transport

let m_ppot_loss (prototypes : Tensor.t) (target_features : Tensor.t) (alpha : float) : Tensor.t =
  let cost_matrix = Optimal_transport.compute_cost_matrix prototypes target_features in
  let transport_plan = Optimal_transport.solve_pot cost_matrix alpha in
  Tensor.sum (Tensor.mul cost_matrix transport_plan)

let reweighted_entropy_loss (logits : Tensor.t) (weights : Tensor.t) : Tensor.t =
  let probs = Tensor.softmax logits ~dim:1 in
  let log_probs = Tensor.log_softmax logits ~dim:1 in
  let entropy = Tensor.sum (Tensor.mul probs log_probs) ~dim:1 in
  Tensor.mean (Tensor.mul weights entropy)

let reweighted_cross_entropy_loss (logits : Tensor.t) (labels : Tensor.t) (weights : Tensor.t) : Tensor.t =
  let loss = Tensor.cross_entropy_loss logits labels ~reduction:Reduction.None in
  Tensor.mean (Tensor.mul weights loss)

let compute_loss (m_ppot : Tensor.t) (entropy : Tensor.t) (cross_entropy : Tensor.t) (config : config) : loss =
  let total = Tensor.(
    (cross_entropy * (Scalar.f config.eta_1)) + 
    (entropy * (Scalar.f config.eta_2)) + 
    (m_ppot * (Scalar.f config.eta_3))
  ) in
  { m_ppot; entropy; cross_entropy; total }