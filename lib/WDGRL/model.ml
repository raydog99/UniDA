open Torch

module FeatureExtractor = struct
  type t = {
    network : nn
  }

  let create input_dim hidden_dim =
    let network = 
      Sequential.([
        linear ~in_features:input_dim ~out_features:hidden_dim ();
        relu ();
        linear ~in_features:hidden_dim ~out_features:hidden_dim ();
        relu ()
      ])
    in
    { network }

  let forward t x = 
    nn_apply t.network x
end

module DomainCritic = struct
  type t = {
    network : nn
  }

  let create hidden_dim =
    let network =
      Sequential.([
        linear ~in_features:hidden_dim ~out_features:hidden_dim ();
        relu ();
        linear ~in_features:hidden_dim ~out_features:1 ()
      ])
    in
    { network }

  let forward t x =
    nn_apply t.network x
end

module Discriminator = struct
  type t = {
    network : nn
  }

  let create hidden_dim num_classes =
    let network =
      Sequential.([
        linear ~in_features:hidden_dim ~out_features:hidden_dim ();
        relu ();
        linear ~in_features:hidden_dim ~out_features:num_classes ()
      ])
    in
    { network }

  let forward t x =
    nn_apply t.network x
end