open Torch

type dataset = {
	x : Tensor.t;
	y : Tensor.t;
}

val create_dataset : num_samples:int -> dim:int -> noise_scale:float -> dataset
val data_loader : dataset -> batch_size:int -> shuffle:bool -> Dataset.t
val split_dataset : dataset -> train_ratio:float -> dataset * dataset