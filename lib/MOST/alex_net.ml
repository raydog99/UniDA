open Torch

module AlexNet = struct
  type t = {
    features: Nn.t;
    avgpool: Nn.t;
    classifier: Nn.t;
  }

  let create num_classes =
    let features = Nn.sequential
      [ Nn.conv2d ~in_channels:3 ~out_channels:64 ~kernel_size:11 ~stride:4 ~padding:2 ();
        Nn.relu ();
        Nn.max_pool2d ~kernel_size:3 ~stride:2 ();
        Nn.conv2d ~in_channels:64 ~out_channels:192 ~kernel_size:5 ~padding:2 ();
        Nn.relu ();
        Nn.max_pool2d ~kernel_size:3 ~stride:2 ();
        Nn.conv2d ~in_channels:192 ~out_channels:384 ~kernel_size:3 ~padding:1 ();
        Nn.relu ();
        Nn.conv2d ~in_channels:384 ~out_channels:256 ~kernel_size:3 ~padding:1 ();
        Nn.relu ();
        Nn.conv2d ~in_channels:256 ~out_channels:256 ~kernel_size:3 ~padding:1 ();
        Nn.relu ();
        Nn.max_pool2d ~kernel_size:3 ~stride:2 () ]
    in
    let avgpool = Nn.adaptive_avg_pool2d ~output_size:[6; 6] in
    let classifier = Nn.sequential
      [ Nn.dropout ~p:0.5 ();
        Nn.linear ~in_features:256*6*6 ~out_features:4096 ();
        Nn.relu ();
        Nn.dropout ~p:0.5 ();
        Nn.linear ~in_features:4096 ~out_features:4096 ();
        Nn.relu ();
        Nn.linear ~in_features:4096 ~out_features:num_classes () ]
    in
    { features; avgpool; classifier }

  let forward t x =
    let x = Nn.apply t.features x in
    let x = Nn.apply t.avgpool x in
    let x = Tensor.flatten x ~start_dim:1 in
    Nn.apply t.classifier x
end

module OptimalTransport = struct
  type t = {
    alexnet: AlexNet.t;
    optimizer: Optimizer.t;
  }

  let create num_classes learning_rate =
    let alexnet = AlexNet.create num_classes in
    let params = List.concat [
      Nn.parameters alexnet.features;
      Nn.parameters alexnet.classifier
    ] in
    let optimizer = Optimizer.adam params ~lr:learning_rate in
    { alexnet; optimizer }

  let wasserstein_loss pred target =
    Tensor.(mean (abs (pred - target)))

  let train_step t source_batch target_batch =
    Optimizer.zero_grad t.optimizer;
    let source_features = AlexNet.forward t.alexnet source_batch in
    let target_features = AlexNet.forward t.alexnet target_batch in
    let loss = wasserstein_loss source_features target_features in
    Tensor.backward loss;
    Optimizer.step t.optimizer;
    loss

  let train t source_data target_data num_iterations batch_size =
    for i = 1 to num_iterations do
      let source_batch = Tensor.narrow source_data ~dim:0 ~start:(i * batch_size mod (Tensor.shape source_data).(0)) ~length:batch_size in
      let target_batch = Tensor.narrow target_data ~dim:0 ~start:(i * batch_size mod (Tensor.shape target_data).(0)) ~length:batch_size in
      let loss = train_step t source_batch target_batch in
      if i mod 100 = 0 then
        Printf.printf "Iteration %d, Loss: %f\n" i (Tensor.to_float0_exn loss)
    done

  let compute_optimal_transport t source_data target_data =
    let source_features = AlexNet.forward t.alexnet source_data in
    let target_features = AlexNet.forward t.alexnet target_data in
    Tensor.cosine_similarity source_features target_features ~dim:1 ~eps:1e-8
end