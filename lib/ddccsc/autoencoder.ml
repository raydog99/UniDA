open Torch
open Utils

type autoencoder = {
  encoder: Tensor.t -> Tensor.t;
  decoder: Tensor.t -> Tensor.t;
}

let linear_relu input_dim output_dim =
  let linear = Layer.linear input_dim output_dim in
  fun x -> Tensor.(relu (linear x))

let init_encoder input_dim hidden_dims =
  check_positive_float (float_of_int input_dim) "input_dim";
  List.iter (fun dim -> check_positive_float (float_of_int dim) "hidden_dim") hidden_dims;
  let layers = 
    List.fold_left
      (fun (prev_dim, acc) dim -> 
        (dim, linear_relu prev_dim dim :: acc))
      (input_dim, [])
      hidden_dims
  in
  let layers = List.rev (snd layers) in
  fun x ->
    check_dimensions x [input_dim] "Invalid input dimensions for encoder";
    List.fold_left (fun acc layer -> layer acc) x layers

let init_decoder output_dim hidden_dims =
  check_positive_float (float_of_int output_dim) "output_dim";
  List.iter (fun dim -> check_positive_float (float_of_int dim) "hidden_dim") hidden_dims;
  let layers = 
    List.fold_left
      (fun (prev_dim, acc) dim -> 
        (dim, linear_relu prev_dim dim :: acc))
      (List.hd hidden_dims, [])
      (List.tl (List.rev hidden_dims) @ [output_dim])
  in
  let layers = List.rev (snd layers) in
  fun x ->
    check_dimensions x [last hidden_dims] "Invalid input dimensions for decoder";
    List.fold_left (fun acc layer -> layer acc) x layers

let create_autoencoder input_dim hidden_dims =
  {
    encoder = init_encoder input_dim hidden_dims;
    decoder = init_decoder input_dim (List.rev hidden_dims);
  }

let reconstruction_loss x x_reconstructed =
  check_dimensions x_reconstructed (Tensor.shape x) "Reconstructed data dimensions do not match input";
  Tensor.mse_loss x x_reconstructed Tensor.Mean

let forward ae x =
  let encoded = ae.encoder x in
  let decoded = ae.decoder encoded in
  (encoded, decoded)