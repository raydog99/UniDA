open Torch
open Types

let encoder (config : config) =
  let open NN in
  Sequential.of_list [
    Linear.create config.input_dim 1024;
    ReLU.create ();
    Linear.create 1024 512;
    ReLU.create ();
    Linear.create 512 config.output_dim;
    Normalize.create ~p:2. ~dim:1;
  ]

let generate_prototypes k d : Tensor.t =
  let m = Tensor.randn [d; k] in
  let q, _ = Tensor.qr m in
  let p = Tensor.mul_scalar q (Float.sqrt (float_of_int k /. (float_of_int k -. 1.))) in
  let i_k = Tensor.eye k in
  let ones = Tensor.ones [k; k] in
  let scaled_ones = Tensor.div_scalar ones (float_of_int k) in
  let diff = Tensor.sub i_k scaled_ones in
  Tensor.mm p diff

let classify (z : Tensor.t) (prototypes : Tensor.t) : Tensor.t =
  let distances = Tensor.cdist z prototypes in
  Tensor.argmin distances ~dim:1 ~keepdim:false

let mse_loss (z_s : Tensor.t) (p_s : Tensor.t) (y_s : Tensor.t) : Tensor.t =
  let p_y = Tensor.index_select p_s ~dim:0 ~index:y_s in
  Tensor.mse_loss z_s p_y ~reduction:Mean

let parametric_cluster_size (tau : Tensor.t) (k_u : int) : Tensor.t =
  let w = Tensor.arange ~end_:(float_of_int k_u) ~options:(T Float, Cuda) in
  let w = Tensor.(pow (add (exp tau) 1.) (div w (float_of_int (k_u - 1)))) in
  Tensor.(div w (sum w))