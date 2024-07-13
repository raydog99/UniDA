open Torch
open Model
open Wd

let train 
    ~feature_extractor ~domain_critic ~discriminator
    ~source_data ~target_data ~num_epochs ~batch_size
    ~critic_iterations ~alpha_1 ~alpha_2 ~lambda ~gamma =
  
  let optimizer_fe = Optimizer.adam (FeatureExtractor.network feature_extractor) ~lr:alpha_2 in
  let optimizer_dc = Optimizer.adam (DomainCritic.network domain_critic) ~lr:alpha_1 in
  let optimizer_disc = Optimizer.adam (Discriminator.network discriminator) ~lr:alpha_2 in

  for epoch = 1 to num_epochs do
    let source_batch, source_labels = sample_batch source_data batch_size in
    let target_batch = sample_batch target_data batch_size in

    for _ = 1 to critic_iterations do
      Optimizer.zero_grad optimizer_dc;
      
      let source_features = FeatureExtractor.forward feature_extractor source_batch in
      let target_features = FeatureExtractor.forward feature_extractor target_batch in
      
      let wd_loss = wasserstein_distance domain_critic source_features target_features in
      let gp_loss = gradient_penalty domain_critic (Tensor.cat [source_features; target_features] ~dim:0) in
      
      let critic_loss = Tensor.(neg wd_loss + (scalar lambda * gp_loss)) in
      Tensor.backward critic_loss;
      Optimizer.step optimizer_dc
    done;

    Optimizer.zero_grad optimizer_fe;
    Optimizer.zero_grad optimizer_disc;

    let source_features = FeatureExtractor.forward feature_extractor source_batch in
    let target_features = FeatureExtractor.forward feature_extractor target_batch in

    let wd_loss = wasserstein_distance domain_critic source_features target_features in
    
    let disc_output = Discriminator.forward discriminator source_features in
    let ce_loss = cross_entropy_loss disc_output source_labels in

    let total_loss = Tensor.(ce_loss + (scalar gamma * wd_loss)) in
    Tensor.backward total_loss;

    Optimizer.step optimizer_fe;
    Optimizer.step optimizer_disc;

    if epoch mod 10 = 0 then
      Printf.printf "Epoch %d: CE Loss: %.4f, WD Loss: %.4f\n" 
        epoch (Tensor.to_float0_exn ce_loss) (Tensor.to_float0_exn wd_loss)
  done