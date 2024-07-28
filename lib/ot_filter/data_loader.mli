open Torch

val create_dataloader : Tensor.t * Tensor.t -> batch_size:int -> (Tensor.t * Tensor.t) Dataset.t
val load_cifar10 : path:string -> (Tensor.t * Tensor.t) * (Tensor.t * Tensor.t)
val load_cifar100 : path:string -> (Tensor.t * Tensor.t) * (Tensor.t * Tensor.t)
val load_clothing1m : path:string -> (Tensor.t * Tensor.t) * (Tensor.t * Tensor.t)
val load_animal10n : path:string -> (Tensor.t * Tensor.t) * (Tensor.t * Tensor.t)