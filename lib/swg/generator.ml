open Torch

type t = {
  model : nn
}

let create input_dim output_dim =
  let model = 
    Nn.sequential
      [
        Nn.linear input_dim 128;
        Nn.relu ();
        Nn.linear 128 256;
        Nn.relu ();
        Nn.linear 256 output_dim;
        Nn.tanh ();
      ]
  in
  { model }

let forward t input =
  Nn.forward t.model input