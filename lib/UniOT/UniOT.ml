open Torch
open Torch

module ResNet50Fc = struct
  type t = {
    conv1: Nn.t;
    bn1: Nn.t;
    layer1: Nn.t;
    layer2: Nn.t;
    layer3: Nn.t;
    layer4: Nn.t;
    fc: Nn.t;
  }
  
  let create pretrained_model_path =
    let conv1 = Nn.conv2d ~in_channels:3 ~out_channels:64 ~kernel_size:7 ~stride:2 ~padding:3 () in
    let bn1 = Nn.batch_norm2d 64 in
    let layer1 = Nn.sequential [Nn.conv2d ~in_channels:64 ~out_channels:64 ~kernel_size:3 ~padding:1 (); Nn.relu ()] in
    let layer2 = Nn.sequential [Nn.conv2d ~in_channels:64 ~out_channels:128 ~kernel_size:3 ~padding:1 (); Nn.relu ()] in
    let layer3 = Nn.sequential [Nn.conv2d ~in_channels:128 ~out_channels:256 ~kernel_size:3 ~padding:1 (); Nn.relu ()] in
    let layer4 = Nn.sequential [Nn.conv2d ~in_channels:256 ~out_channels:512 ~kernel_size:3 ~padding:1 (); Nn.relu ()] in
    let fc = Nn.linear ~in_features:512 ~out_features:2048 () in
    {conv1; bn1; layer1; layer2; layer3; layer4; fc}

  let forward t input =
    let open Tensor in
    input
    |> Nn.apply t.conv1
    |> Nn.apply t.bn1
    |> relu
    |> max_pool2d ~kernel_size:[3; 3] ~stride:[2; 2] ~padding:[1; 1]
    |> Nn.apply t.layer1
    |> Nn.apply t.layer2
    |> Nn.apply t.layer3
    |> Nn.apply t.layer4
    |> adaptive_avg_pool2d ~output_size:[1; 1]
    |> view ~size:[-1; 512]
    |> Nn.apply t.fc

  let output_dim _ = 2048

  let parameters t =
    List.concat [
      Nn.parameters t.conv1;
      Nn.parameters t.bn1;
      Nn.parameters t.layer1;
      Nn.parameters t.layer2;
      Nn.parameters t.layer3;
      Nn.parameters t.layer4;
      Nn.parameters t.fc;
    ]
end

module CLS = struct
  let weight_norm t =
    let normalize_weights w =
      let norm = Tensor.(norm w ~p:2 ~dim:0 ~keepdim:true) in
      Tensor.(div w norm)
    in
    t.fc1 <- normalize_weights t.fc1;
    t.fc2 <- normalize_weights t.fc2
end

module ProtoCLS = struct
  let weight_norm t =
    let norm = Tensor.(norm t.fc ~p:2 ~dim:1 ~keepdim:true) in
    t.fc <- Tensor.(div t.fc norm)
end

module MemoryQueue = struct
  let get_nearest_neighbor t features ids =
    let sim = Tensor.(mm features (transpose t.queue ~dim0:0 ~dim1:1)) in
    let values, indices = Tensor.max sim ~dim:1 in
    let neighbor_feats = Tensor.index_select t.queue ~dim:0 ~index:indices in
    (values, neighbor_feats)

  let random_sample t size =
    let total_size = Tensor.shape t.queue |> List.hd in
    let indices = Tensor.randint ~high:total_size size ~dtype:(T Int64) in
    Tensor.index_select t.queue ~dim:0 ~index:indices
end

module OptimWithSheduler = struct
  let step t ~loss =
    Optimizer.zero_grad t.optimizer;
    Tensor.backward loss;
    let lr = t.scheduler (Optimizer.learning_rate t.optimizer) (Optimizer.step t.optimizer) in
    Optimizer.set_learning_rate t.optimizer lr
end

let sinkhorn input ~epsilon ~sinkhorn_iterations =
  let n = Tensor.shape input |> List.hd in
  let log_input = Tensor.log input in
  let mut_Q = Tensor.copy log_input in
  let sum_Q = ref (Tensor.sum mut_Q ~dim:1 ~keepdim:true) in
  for _ = 1 to sinkhorn_iterations do
    mut_Q <- Tensor.(sub mut_Q (log !sum_Q));
    sum_Q := Tensor.(sum (exp mut_Q) ~dim:0 ~keepdim:true);
    mut_Q <- Tensor.(sub mut_Q (log !sum_Q));
    sum_Q := Tensor.(sum (exp mut_Q) ~dim:1 ~keepdim:true)
  done;
  Tensor.(exp mut_Q)

let adaptive_filling ubot_feature_t source_prototype gamma beta fill_size_uot =
  let sim = Tensor.(mm ubot_feature_t (transpose source_prototype ~dim0:0 ~dim1:1)) in
  let max_sim, _ = Tensor.max sim ~dim:1 in
  let threshold = Tensor.mean max_sim in
  let mask = Tensor.(gt max_sim threshold) in
  let high_conf_size = Tensor.(sum mask ~dim:0 |> to_int0_exn) in
  let fake_size = min high_conf_size fill_size_uot in
  let newsim = Tensor.(mul_scalar sim gamma) in
  (newsim, fake_size)

let ubot_CCD newsim beta ~fake_size ~fill_size ~mode =
  let n, m = Tensor.shape newsim in
  let alpha = Tensor.ones [n] ~kind:(T Float) |> Tensor.div_scalar (float_of_int n) in
  let beta = match beta with
    | None -> Tensor.ones [m] ~kind:(T Float) |> Tensor.div_scalar (float_of_int m)
    | Some b -> b
  in
  let newsim_exp = Tensor.exp newsim in
  let mut_P = Tensor.copy newsim_exp in
  for _ = 1 to 10 do  (* Simplified Sinkhorn-Knopp iteration *)
    mut_P <- Tensor.(div mut_P (sum mut_P ~dim:1 ~keepdim:true));
    mut_P <- Tensor.(div mut_P (sum mut_P ~dim:0 ~keepdim:true))
  done;
  let values, indices = Tensor.max mut_P ~dim:1 in
  let high_conf_label_id = Tensor.masked_select (Tensor.arange n) (Tensor.gt values 0.5) in
  let high_conf_label = Tensor.index_select indices ~dim:0 ~index:high_conf_label_id in
  let new_beta = Tensor.(sum mut_P ~dim:0) |> Tensor.div_scalar (float_of_int n) in
  (high_conf_label_id, high_conf_label, mut_P, new_beta)

let () =
  let cls_output_dim = List.length source_classes in
  let feat_dim = 256 in
  let feature_extractor = ResNet50Fc.create pretrained_model_path in
  let classifier = CLS.create (ResNet50Fc.output_dim feature_extractor) cls_output_dim ~hidden_mlp:2048 ~feat_dim:256 ~temp in
  let cluster_head = ProtoCLS.create feat_dim K ~temp in

  let feature_extractor = Tensor.to_device feature_extractor ~device:Cuda in
  let classifier = Tensor.to_device classifier ~device:Cuda in
  let cluster_head = Tensor.to_device cluster_head ~device:Cuda in

  let optimizer_featex = Optimizer.sgd (ResNet50Fc.parameters feature_extractor) ~lr:(args.train.lr *. 0.1) ~momentum:args.train.sgd_momentum ~weight_decay:args.train.weight_decay ~nesterov:true in
  let optimizer_cls = Optimizer.sgd (CLS.parameters classifier) ~lr:args.train.lr ~momentum:args.train.sgd_momentum ~weight_decay:args.train.weight_decay ~nesterov:true in
  let optimizer_cluhead = Optimizer.sgd (ProtoCLS.parameters cluster_head) ~lr:args.train.lr ~momentum:args.train.sgd_momentum ~weight_decay:args.train.weight_decay ~nesterov:true in

  let scheduler step initial_lr = inverseDecaySheduler step initial_lr ~gamma:10 ~power:0.75 ~max_iter:args.train.min_step in
  let opt_sche_featex = OptimWithSheduler.create optimizer_featex scheduler in
  let opt_sche_cls = OptimWithSheduler.create optimizer_cls scheduler in
  let opt_sche_cluhead = OptimWithSheduler.create optimizer_cluhead scheduler in

  let feature_extractor = Nn.data_parallel feature_extractor in
  let classifier = Nn.data_parallel classifier in
  let cluster_head = Nn.data_parallel cluster_head in

  let target_size = Dataset.length target_train_ds in
  let n_batch = MQ_size / batch_size in
  let memqueue = MemoryQueue.create feat_dim batch_size n_batch ~temp in

  let global_step = ref 0 in
  let beta = ref None in

  while !global_step < args.train.min_step do
    let iters = Iter.zip source_train_dl target_train_dl in
    Iter.iter (fun ((im_source, label_source, id_source), (im_target, _, id_target)) ->
      let label_source = Tensor.to_device label_source ~device:Cuda in
      let im_source = Tensor.to_device im_source ~device:Cuda in
      let im_target = Tensor.to_device im_target ~device:Cuda in

      let feature_ex_s = ResNet50Fc.forward feature_extractor im_source in
      let feature_ex_t = ResNet50Fc.forward feature_extractor im_target in

      let (before_lincls_feat_s, after_lincls_s) = CLS.forward classifier feature_ex_s in
      let (before_lincls_feat_t, after_lincls_t) = CLS.forward classifier feature_ex_t in

      let norm_feat_s = Tensor.normalize before_lincls_feat_s ~p:2 ~dim:1 in
      let norm_feat_t = Tensor.normalize before_lincls_feat_t ~p:2 ~dim:1 in

      let after_cluhead_t = ProtoCLS.forward cluster_head before_lincls_feat_t in

      let criterion = Nn.cross_entropy_loss () in
      let loss_cls = criterion after_lincls_s label_source in

      let minibatch_size = Tensor.shape norm_feat_t |> List.hd in

      let loss_all = Tensor.(add loss_cls (mul_scalar (add loss_PCD loss_CCD) lam)) in

      Optimizer.backward_step ~loss:loss_all [opt_sche_featex; opt_sche_cls; opt_sche_cluhead];

      CLS.weight_norm classifier;
      ProtoCLS.weight_norm cluster_head;
      MemoryQueue.update_queue memqueue norm_feat_t (Tensor.to_device id_target ~device:Cuda);
      incr global_step
    ) iters
  done