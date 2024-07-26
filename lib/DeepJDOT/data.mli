open Torch

type dataset = private {
  images: Tensor.t;
  labels: Tensor.t;
}

val load_mnist : unit -> dataset

val create_dataloader : dataset -> int -> bool -> int * (int -> Tensor.t * Tensor.t)