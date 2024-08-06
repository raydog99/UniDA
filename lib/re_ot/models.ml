open Torch

type t = {
  feature_extractor: NN.t;
  classifier: NN.t;
  mutable q: Tensor.t;
}

let create input_dim hidden_dim num_classes =
  let feature_extractor = NN.sequential
    [
      NN.linear input_dim hidden_dim;
      NN.relu;
      NN.linear hidden_dim hidden_dim;
      NN.relu;
    ]
  in
  let classifier = NN.linear hidden_dim num_classes in
  let q = Tensor.ones [|num_classes; num_classes|] in
  { feature_extractor; classifier; q }

let forward t x =
  let features = NN.forward t.feature_extractor x in
  NN.forward t.classifier features

let parameters t =
  NN.Parameters.concat [NN.parameters t.feature_extractor; NN.parameters t.classifier]

let update_q t new_q =
  t.q <- new_q