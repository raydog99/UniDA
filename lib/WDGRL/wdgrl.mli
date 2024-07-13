open Torch
open Model

val train : 
  feature_extractor:FeatureExtractor.t ->
  domain_critic:DomainCritic.t ->
  discriminator:Discriminator.t ->
  source_data:Tensor.t ->
  target_data:Tensor.t ->
  num_epochs:int ->
  batch_size:int ->
  critic_iterations:int ->
  alpha_1:float ->
  alpha_2:float ->
  lambda:float ->
  gamma:float ->
  unit