open Torch
open Types

module RobustOT = struct
  let robust_wasserstein_dual p_x p_y rho1 rho2 max_iter learning_rate =
    let open Tensor in
    let f t = (t - one) * (t - one) / (Scalar.float 2.) in
    let optimize_weights w_x w_y d_x d_y =
      let obj = dot w_x d_x - dot w_y d_y in
      let constraint_x = mean (f (w_x * (float (shape w_x |> List.hd)))) - Scalar.float rho1 in
      let constraint_y = mean (f (w_y * (float (shape w_y |> List.hd)))) - Scalar.float rho2 in
      obj + relu constraint_x + relu constraint_y
    in
    let rec optimize_loop w_x w_y d iter =
      if iter >= max_iter then (w_x, w_y)
      else
        let d_x = d p_x in
        let d_y = d p_y in
        let loss = optimize_weights w_x w_y d_x d_y in
        let grad_w_x, grad_w_y = grad_of_fn2 optimize_weights w_x w_y in
        let w_x' = w_x - grad_w_x * (Scalar.float learning_rate) in
        let w_y' = w_y - grad_w_y * (Scalar.float learning_rate) in
        optimize_loop w_x' w_y' d (iter + 1)
    in
    let initial_w_x = ones (shape p_x) in
    let initial_w_y = ones (shape p_y) in
    let d = Layer.linear ~input_dim:(shape p_x |> List.hd) ~output_dim:1 in
    optimize_loop initial_w_x initial_w_y d 0

  let discrete_formulation x y rho1 rho2 max_iter learning_rate =
    robust_wasserstein_dual x y rho1 rho2 max_iter learning_rate

  let continuous_relaxation p_x p_y rho1 rho2 max_iter learning_rate =
    let open Tensor in
    let f t = (t - one) * (t - one) / (Scalar.float 2.) in
    let optimize_step w_x w_y d =
      let obj = mean (w_x * d p_x) - mean (w_y * d p_y) in
      let constraint_x = relu (mean (f w_x) - Scalar.float rho1) in
      let constraint_y = relu (mean (f w_y) - Scalar.float rho2) in
      obj + constraint_x + constraint_y
    in
    let rec optimize_loop w_x w_y d iter =
      if iter >= max_iter then (w_x, w_y)
      else
        let loss = optimize_step w_x w_y d in
        let grad_w_x, grad_w_y = grad_of_fn2 optimize_step w_x w_y in
        let w_x' = w_x - grad_w_x * (Scalar.float learning_rate) in
        let w_y' = w_y - grad_w_y * (Scalar.float learning_rate) in
        optimize_loop w_x' w_y' d (iter + 1)
    in
    let initial_w_x = ones (shape p_x) in
    let initial_w_y = ones (shape p_y) in
    let d = Layer.linear ~input_dim:(shape p_x |> List.hd) ~output_dim:1 in
    optimize_loop initial_w_x initial_w_y d 0

  let gan_optimization
      (type a b c)
      (module G : Network with type t = a)
      (module D : Network with type t = b)
      (module W : Network with type t = c)
      (module O : Optimizer)
      generator discriminator w_network p_x rho lambda max_iter learning_rate =
    let open Tensor in
    let optimize_step real_data fake_data =
      let w_x = W.forward w_network real_data in
      let d_real = D.forward discriminator real_data in
      let d_fake = D.forward discriminator fake_data in
      let gan_loss = mean (w_x * d_real) - mean d_fake in
      let constraint = relu (mean ((w_x - one) * (w_x - one) / (Scalar.float 2.)) - Scalar.float rho) in
      gan_loss + Scalar.float lambda * constraint
    in
    let rec optimize_loop optimizer iter =
      if iter >= max_iter then ()
      else
        let real_data = p_x () in
        let noise = Tensor.randn [Tensor.shape real_data |> List.hd; G.input_dim generator] in
        let fake_data = G.forward generator noise in
        let loss = optimize_step real_data fake_data in
        O.zero_grad optimizer;
        Tensor.backward loss;
        O.step optimizer;
        optimize_loop optimizer (iter + 1)
    in
    let optimizer = O.create (G.parameters generator @ D.parameters discriminator @ W.parameters w_network) learning_rate in
    optimize_loop optimizer 0

  let domain_adaptation_optimization
      (type a b c d)
      (module F : Network with type t = a)
      (module C : Network with type t = b)
      (module W : Network with type t = c)
      (module O : Optimizer)
      feature_network classifier w_network source_data target_data rho max_iter learning_rate =
    let open Tensor in
    let optimize_step source_x source_y target_x =
      let source_features = F.forward feature_network source_x in
      let target_features = F.forward feature_network target_x in
      let source_pred = C.forward classifier source_features in
      let target_pred = C.forward classifier target_features in
      let classification_loss = cross_entropy source_pred source_y in
      let w_t = W.forward w_network target_features in
      let domain_loss = mean (w_t * target_pred) - mean source_pred in
      let constraint = relu (mean ((w_t - one) * (w_t - one) / (Scalar.float 2.)) - Scalar.float rho) in
      classification_loss + domain_loss + constraint
    in
    let rec optimize_loop optimizer iter =
      if iter >= max_iter then ()
      else
        let source_x, source_y = source_data () in
        let target_x = target_data () in
        let loss = optimize_step source_x source_y target_x in
        O.zero_grad optimizer;
        Tensor.backward loss;
        O.step optimizer;
        optimize_loop optimizer (iter + 1)
    in
    let optimizer = O.create (F.parameters feature_network @ C.parameters classifier @ W.parameters w_network) learning_rate in
    optimize_loop optimizer 0
end

module Generator : Network = struct
  type t = {
    model : Layer.t;
    input_dim : int;
  }

  let create input_dim hidden_dim output_dim =
    let model = Layer.sequential [
      Layer.linear ~input_dim ~output_dim:hidden_dim;
      Layer.relu;
      Layer.linear ~input_dim:hidden_dim ~output_dim;
      Layer.tanh;
    ] in
    { model; input_dim }

  let forward t input = Layer.forward t.model input
  let parameters t = Layer.parameters t.model
  let input_dim t = t.input_dim
end

module Discriminator : Network = struct
  type t = {
    model : Layer.t;
  }

  let create input_dim hidden_dim output_dim =
    let model = Layer.sequential [
      Layer.linear ~input_dim ~output_dim:hidden_dim;
      Layer.relu;
      Layer.linear ~input_dim:hidden_dim ~output_dim;
    ] in
    { model }

  let forward t input = Layer.forward t.model input
  let parameters t = Layer.parameters t.model
end

module WeightNetwork : Network = struct
  type t = {
    model : Layer.t;
  }

  let create input_dim hidden_dim output_dim =
    let model = Layer.sequential [
      Layer.linear ~input_dim ~output_dim:hidden_dim;
      Layer.relu;
      Layer.linear ~input_dim:hidden_dim ~output_dim;
      Layer.relu;
    ] in
    { model }

  let forward t input =
    let weights = Layer.forward t.model input in
    Tensor.(weights / sum weights)

  let parameters t = Layer.parameters t.model
end

module FeatureNetwork : Network = struct
  type t = {
    model : Layer.t;
  }

  let create input_dim hidden_dim output_dim =
    let model = Layer.sequential [
      Layer.linear ~input_dim ~output_dim:hidden_dim;
      Layer.relu;
      Layer.linear ~input_dim:hidden_dim ~output_dim;
    ] in
    { model }

  let forward t input = Layer.forward t.model input
  let parameters t = Layer.parameters t.model
end

module Classifier : Network = struct
  type t = {
    model : Layer.t;
  }

  let create input_dim hidden_dim num_classes =
    let model = Layer.sequential [
      Layer.linear ~input_dim ~output_dim:hidden_dim;
      Layer.relu;
      Layer.linear ~input_dim:hidden_dim ~output_dim:num_classes;
    ] in
    { model }

  let forward t input = Layer.forward t.model input
  let parameters t = Layer.parameters t.model
end

module Optimizer : Optimizer = struct
  type t = Torch.Optimizer.t

  let create parameters learning_rate =
    Optimizer.adam parameters ~learning_rate

  let step = Optimizer.step
  let zero_grad = Optimizer.zero_grad
end