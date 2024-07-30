open Torch

type config = {
  num_classes : int;
  batch_size : int;
  learning_rate : float;
  alpha : float;
  beta : float;
  momentum : float;
  eta_1 : float;
  eta_2 : float;
  eta_3 : float;
  tau_1 : float;
  tau_2 : float;
  num_epochs : int;
}

type model = {
  feature_extractor : Nn.t;
  classifier : Nn.t;
}

type loss = {
  m_ppot : Tensor.t;
  entropy : Tensor.t;
  cross_entropy : Tensor.t;
  total : Tensor.t;
}

type training_state = {
  model : model;
  optimizer : Optimizer.t;
  mutable alpha : float;
  mutable beta : float;
}