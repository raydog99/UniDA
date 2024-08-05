open Torch
open Types
open Model
open Sinkhorn
open Buffer

let train (config : config) (encoder : NN.t) (prototypes : Tensor.t) (known_data : dataset) (novel_data : dataset) =
  let optimizer = Optimizer.adam (Module.parameters encoder) ~lr:config.learning_rate in
  let num_novel_classes = Tensor.shape prototypes |> List.nth 1 in
  let tau = Tensor.zeros [] ~requires_grad:true in
  let buffer = Buffer.create config.buffer_size num_novel_classes in

  for epoch = 1 to config.num_epochs do
    let x_s, y_s = Data.sample_batch known_data config.batch_size in
    let x_u = Data.sample_batch novel_data config.batch_size |> fst in

    let z_s = NN.forward encoder x_s in
    let z_u = NN.forward encoder x_u in

    let l_s = mse_loss z_s prototypes y_s in
    let w = parametric_cluster_size tau num_novel_classes in
    let l_u = adaptive_self_labeling_loss z_u prototypes w config.gamma buffer in

    let loss = Tensor.(add l_s (mul_scalar l_u 0.5)) in

    Optimizer.zero_grad optimizer;
    Tensor.backward loss;
    Optimizer.step optimizer;

    let tau_optimizer = Optimizer.adam [tau] ~lr:0.01 in
    Optimizer.step tau_optimizer;

    if epoch mod 10 = 0 then
      Printf.printf "Epoch %d, Loss: %.4f\n" epoch (Tensor.to_float0_exn loss)
  done;
  encoder