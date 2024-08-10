open Torch

type t = {
  generator : Generator.t;
  discriminator : Discriminator.t;
  latent_dim : int;
  device : Device.t;
}

type config = {
  latent_dim : int;
  data_dim : int;
  learning_rate_g : float;
  learning_rate_d : float;
  beta1 : float;
  beta2 : float;
}

let create config device =
  {
    generator = Generator.create config.latent_dim config.data_dim;
    discriminator = Discriminator.create config.data_dim;
    latent_dim = config.latent_dim;
    device;
  }

let to_device t =
  let generator = Nn.to_device t.generator.model t.device in
  let discriminator_fe = Nn.to_device t.discriminator.feature_extractor t.device in
  let discriminator_fl = Nn.to_device t.discriminator.final_layer t.device in
  { t with 
    generator = { model = generator };
    discriminator = { feature_extractor = discriminator_fe; final_layer = discriminator_fl };
  }

let generate t batch_size =
  let noise = Tensor.randn [batch_size; t.latent_dim] ~device:t.device in
  Generator.forward t.generator noise

let discriminate t input =
  Discriminator.forward t.discriminator input

let wasserstein_distance real_proj fake_proj =
  let sorted_real = Tensor.sort real_proj ~descending:false |> fst in
  let sorted_fake = Tensor.sort fake_proj ~descending:false |> fst in
  Tensor.mse_loss sorted_real sorted_fake

let max_sliced_wasserstein_distance real_features fake_features =
  let feature_dim = Tensor.shape real_features |> List.nth 1 in
  
  let num_projections = 1000 in
  let projections = Tensor.randn [num_projections; feature_dim] ~device:real_features.device in
  let projections = Tensor.div projections (Tensor.norm projections ~dim:[1] ~keepdim:true) in
  
  let real_projections = Tensor.matmul real_features projections ~transpose_b:true in
  let fake_projections = Tensor.matmul fake_features projections ~transpose_b:true in
  
  let distances = Tensor.zeros [num_projections] ~device:real_features.device in
  for i = 0 to num_projections - 1 do
    let real_proj = Tensor.select real_projections ~dim:1 ~index:i in
    let fake_proj = Tensor.select fake_projections ~dim:1 ~index:i in
    let dist = wasserstein_distance real_proj fake_proj in
    Tensor.set distances [i] dist
  done;
  
  Tensor.max distances |> fst

let train_step t real_data optimizer_g optimizer_d =
  let batch_size = Tensor.shape real_data |> List.hd in
  
  Optimizer.zero_grad optimizer_d;
  let fake_data = generate t batch_size in
  let real_features, real_output = discriminate t real_data in
  let fake_features, fake_output = discriminate t fake_data in
  
  let d_loss = Discriminator.surrogate_loss real_output fake_output in
  Tensor.backward d_loss;
  Optimizer.step optimizer_d;
  
  Optimizer.zero_grad optimizer_g;
  let fake_data = generate t batch_size in
  let fake_features, _ = discriminate t fake_data in
  
  let g_loss = max_sliced_wasserstein_distance real_features fake_features in
  Tensor.backward g_loss;
  Optimizer.step optimizer_g;

  (Tensor.float_value g_loss, Tensor.float_value d_loss)

let evaluate t data_loader =
  let total_loss = ref 0. in
  let num_batches = ref 0 in
  Data_loader.iter data_loader ~f:(fun batch ->
    let real_data = Tensor.to_device batch t.device in
    let batch_size = Tensor.shape real_data |> List.hd in
    let fake_data = generate t batch_size in
    let real_features, _ = discriminate t real_data in
    let fake_features, _ = discriminate t fake_data in
    let loss = max_sliced_wasserstein_distance real_features fake_features in
    total_loss := !total_loss +. (Tensor.float_value loss);
    num_batches := !num_batches + 1
  );
  !total_loss /. (float_of_int !num_batches)

let save t filename =
  let state_dict = [
    ("generator", Nn.state_dict t.generator.model);
    ("discriminator_fe", Nn.state_dict t.discriminator.feature_extractor);
    ("discriminator_fl", Nn.state_dict t.discriminator.final_layer);
  ] in
  Serialize.save ~filename state_dict

let load config device filename =
  let state_dict = Serialize.load ~filename in
  let t = create config device in
  let generator = Nn.load_state_dict t.generator.model (List.assoc "generator" state_dict) in
  let discriminator_fe = Nn.load_state_dict t.discriminator.feature_extractor (List.assoc "discriminator_fe" state_dict) in
  let discriminator_fl = Nn.load_state_dict t.discriminator.final_layer (List.assoc "discriminator_fl" state_dict) in
  { t with 
    generator = { model = generator };
    discriminator = { feature_extractor = discriminator_fe; final_layer = discriminator_fl };
  }