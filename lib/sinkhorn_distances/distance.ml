open Torch

let l2_squared_distance (x : Tensor.t) (y : Tensor.t) =
  let x_norm = Tensor.sum (Tensor.pow x 2.) 1 in
  let y_norm = Tensor.sum (Tensor.pow y 2.) 1 in
  let xy = Tensor.mm x (Tensor.transpose y ~dim0:(-1) ~dim1:(-2)) in
  Tensor.add (Tensor.add (Tensor.reshape x_norm [-1; 1]) (Tensor.reshape y_norm [1; -1])) (Tensor.mul_scalar xy (-2.))

let cosine_distance (x : Tensor.t) (y : Tensor.t) =
  let x_norm = Tensor.sqrt (Tensor.sum (Tensor.pow x 2.) 1) in
  let y_norm = Tensor.sqrt (Tensor.sum (Tensor.pow y 2.) 1) in
  let xy = Tensor.mm x (Tensor.transpose y ~dim0:(-1) ~dim1:(-2)) in
  Tensor.sub (Tensor.ones [Tensor.size xy 0; Tensor.size xy 1]) 
    (Tensor.div xy (Tensor.mm (Tensor.reshape x_norm [-1; 1]) (Tensor.reshape y_norm [1; -1])))