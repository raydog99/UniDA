open Torch

type dataset = {
  images: Tensor.t;
  labels: Tensor.t;
}

type config = {
  input_dim: int;
  output_dim: int;
  num_epochs: int;
  batch_size: int;
  learning_rate: float;
  gamma: float;
  beta: float;
  max_k_u: int;
  buffer_size: int;
}