open Torch

type autoencoder = {
  encoder: Tensor.t -> Tensor.t;
  decoder: Tensor.t -> Tensor.t;
}

val create_autoencoder : int -> int list -> autoencoder
val reconstruction_loss : Tensor.t -> Tensor.t -> Tensor.t
val forward : autoencoder -> Tensor.t -> Tensor.t * Tensor.t