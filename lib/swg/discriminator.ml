open Torch

type t = {
  feature_extractor : nn;
  final_layer : nn;
}

let create input_dim =
  let feature_extractor = 
    Nn.sequential
      [
        Nn.linear input_dim 256;
        Nn.leaky_relu ();
        Nn.linear 256 128;
        Nn.leaky_relu ();
      ]
  in
  let final_layer = Nn.linear 128 1 in
  { feature_extractor; final_layer }

let forward t input =
  let features = Nn.forward t.feature_extractor input in
  let output = Nn.forward t.final_layer features in
  (features, output)

let surrogate_loss real_output fake_output =
  let real_loss = Tensor.binary_cross_entropy_with_logits 
    ~input:real_output 
    ~target:(Tensor.ones_like real_output) in
  let fake_loss = Tensor.binary_cross_entropy_with_logits
    ~input:fake_output
    ~target:(Tensor.zeros_like fake_output) in
  Tensor.(real_loss + fake_loss)