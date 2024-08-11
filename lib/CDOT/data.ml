open Base
open Torch

type dataset = {
  features: Tensor.t;
  labels: Tensor.t;
}

let create_rotating_moons n_samples angle =
  let t = Tensor.linspace ~start:0.0 ~end_:Float.pi ~steps:n_samples in
  let x = Tensor.cos t in
  let y = Tensor.sin t in
  let x = Tensor.mul_scalar x 0.7 in
  let y = Tensor.sub y (Tensor.abs (Tensor.div_scalar x 2.0)) in
  let features = Tensor.stack [x; y] ~dim:1 in
  let features = Tensor.cat [features; Tensor.neg features] ~dim:0 in
  let labels = Tensor.cat [Tensor.zeros [n_samples]; Tensor.ones [n_samples]] ~dim:0 in
  let rotation_matrix = Tensor.of_2d_list [[Float.cos angle; -.Float.sin angle]; [Float.sin angle; Float.cos angle]] in
  let rotated_features = Tensor.matmul features rotation_matrix in
  { features = rotated_features; labels }

let create_source_target_seq n_source n_target n_steps angle_step =
  let source = create_rotating_moons n_source 0.0 in
  let targets = List.init n_steps ~f:(fun i ->
    create_rotating_moons n_target (Float.of_int (i + 1) *. angle_step)
  ) in
  source, targets