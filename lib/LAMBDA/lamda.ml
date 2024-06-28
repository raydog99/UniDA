open Torch

module Generator = struct
  let create () = Nn.sequential [
    Nn.linear ~in_features:100 ~out_features:256 ();
    Nn.relu ();
    Nn.linear ~in_features:256 ~out_features:512 ();
    Nn.relu ();
    Nn.linear ~in_features:512 ~out_features:1024 ();
    Nn.tanh ()
  ]
end

module Classifier = struct
  let create () = Nn.sequential [
    Nn.linear ~in_features:1024 ~out_features:512 ();
    Nn.relu ();
    Nn.linear ~in_features:512 ~out_features:256 ();
    Nn.relu ();
    Nn.linear ~in_features:256 ~out_features:1 ();
    Nn.sigmoid ()
  ]
end

module LAMDA = struct
  type t = {
    g1 : Nn.t;
    t1 : Nn.t;
    t2 : Nn.t;
    a : Nn.t;
    optimizer : Optimizer.t;
  }

  let create () =
    let g1 = Generator.create () in
    let t1 = Generator.create () in
    let t2 = Generator.create () in
    let a = Classifier.create () in
    let params = List.concat [Nn.parameters g1; Nn.parameters t1; Nn.parameters t2; Nn.parameters a] in
    let optimizer = Optimizer.adam params ~lr:0.0002 in
    { g1; t1; t2; a; optimizer }

  let train_step t source_batch target_batch =
    Optimizer.zero_grad t.optimizer;
    let loss = Tensor.(
      mean (binary_cross_entropy 
        (t.a (t.g1 source_batch))
        (full_like target_batch 1.)
      ) +
      mean (binary_cross_entropy
        (t.a (t.t1 (t.t2 target_batch)))
        (full_like target_batch 0.)
      )
    ) in
    Tensor.backward loss;
    Optimizer.step t.optimizer;
    loss

  let train t source_data target_data num_iterations batch_size =
    for _ = 1 to num_iterations do
      let source_batch = Tensor.slice source_data ~dim:0 ~start:0 ~end_:batch_size in
      let target_batch = Tensor.slice target_data ~dim:0 ~start:0 ~end_:batch_size in
      let loss = train_step t source_batch target_batch in
      Stdio.printf "Loss: %f\n" (Tensor.to_float0_exn loss)
    done

endd