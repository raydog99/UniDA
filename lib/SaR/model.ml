open Base
open Torch

type t = {
  encoder : Module.t;
  classifier : Module.t;
}

let create input_dim num_classes =
  let encoder = Module.sequential
    [
      Module.conv2d ~input_channels:3 ~output_channels:16 ~kernel_size:3 ~stride:1 ~padding:1 ();
      Module.batch_norm2d 16;
      Module.relu ();
      Module.max_pool2d ~kernel_size:2 ();
      Module.conv2d ~input_channels:16 ~output_channels:32 ~kernel_size:3 ~stride:1 ~padding:1 ();
      Module.batch_norm2d 32;
      Module.relu ();
      Module.max_pool2d ~kernel_size:2 ();
      Module.flatten ();
    ] in
  let classifier = Module.linear ~input_dim ~output_dim:num_classes () in
  { encoder; classifier }

let forward t x =
  let features = Module.forward t.encoder x in
  Module.forward t.classifier features

let predict t x =
  Tensor.softmax (forward t x) ~dim:1

let supervised_loss t x y =
  let predictions = forward t x in
  Tensor.cross_entropy_loss predictions y ~reduction:Mean

let unsupervised_loss t x y_pseudo =
  let predictions = predict t x in
  Tensor.mse_loss predictions y_pseudo ~reduction:Mean