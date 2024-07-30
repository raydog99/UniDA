open Torch

let create (in_features : int) (num_classes : int) : Nn.t =
  Nn.Sequential.of_list [
    Nn.Linear.create ~in_features ~out_features:256;
    Nn.Relu.create ();
    Nn.Linear.create ~in_features:256 ~out_features:num_classes
  ]

let classify (classifier : Nn.t) (features : Tensor.t) : Tensor.t =
  Nn.Module.forward classifier features