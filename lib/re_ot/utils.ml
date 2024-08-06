open Torch

module Tensor = Torch.Tensor

let softmax tensor axis =
  let max_vals = Tensor.max tensor ~dim:[axis] ~keepdim:true in
  let exp_vals = Tensor.exp (Tensor.sub tensor max_vals) in
  let sum_exp = Tensor.sum exp_vals ~dim:[axis] ~keepdim:true in
  Tensor.div exp_vals sum_exp

let kl_divergence p q =
  let p = Tensor.clip p ~min:1e-8 ~max:1.0 in
  let q = Tensor.clip q ~min:1e-8 ~max:1.0 in
  Tensor.sum (Tensor.mul p (Tensor.log (Tensor.div p q)))

let cosine_distance x y =
  let norm_x = Tensor.norm x ~p:2 ~dim:[1] ~keepdim:true in
  let norm_y = Tensor.norm y ~p:2 ~dim:[1] ~keepdim:true in
  let dot_product = Tensor.matmul x (Tensor.transpose y ~dim0:0 ~dim1:1) in
  Tensor.div dot_product (Tensor.matmul norm_x (Tensor.transpose norm_y ~dim0:0 ~dim1:1))