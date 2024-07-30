open Torch
open Torch_vision

let create () : Nn.t =
  Resnet.resnet50 ~pretrained:true ()

let extract_features (extractor : Nn.t) (input : Tensor.t) : Tensor.t =
  let features = Nn.Module.forward extractor input in
  Tensor.view features ~size:[-1; 2048]