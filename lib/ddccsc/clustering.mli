open Torch
open Autoencoder

type params = {
  n_clusters: int;
  epsilon: float;
  n_iter: int;
  learning_rate: float;
  n_epochs: int;
}

val cluster : Tensor.t -> params -> Autoencoder.autoencoder * Tensor.t
val assign_clusters : Autoencoder.autoencoder -> Tensor.t -> Tensor.t -> Tensor.t