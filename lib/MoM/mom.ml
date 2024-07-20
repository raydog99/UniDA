open Torch

let create_blocks tensor k =
  let n = Tensor.shape tensor |> List.hd in
  let block_size = n / k in
  List.init k (fun i ->
    Tensor.narrow tensor ~dim:0 ~start:(i * block_size) ~length:block_size)

let median_block blocks =
  let values = List.map (fun block -> Tensor.mean block) blocks in
  let sorted = List.sort Tensor.compare values in
  List.nth sorted (List.length sorted / 2)

let estimate x y k_x k_y learning_rate =
  let n_iter = 1000 in
  let c = 0.01 in
  
  let w = Tensor.randn [Tensor.shape x |> List.hd; 1] in
  
  let rec train_loop w t =
    if t >= n_iter then w
    else
      let x_blocks = create_blocks x k_x in
      let y_blocks = create_blocks y k_y in
      let x_med = median_block x_blocks in
      let y_med = median_block y_blocks in
      
      let grad = Tensor.(x_med - y_med) in
      let w' = Tensor.(w + grad * f learning_rate) in
      let w'' = Tensor.clamp w' ~min:(Float.neg c) ~max:c in
      
      train_loop w'' (t + 1)
  in
  
  train_loop w 0