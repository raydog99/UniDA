open Torch
open Estimator.Utils

let create_blocks = create_blocks
let median_block = median_block

let estimate x y k_x k_y learning_rate n_iter =
  let c = 0.01 in
  let w = Tensor.randn [Tensor.shape x |> List.tl |> List.hd; 1] in
  
  let rec train_loop w t =
    if t >= n_iter then w
    else
      let x_blocks = create_blocks x k_x in
      let y_blocks = create_blocks y k_y in
      
      let block_estimates = 
        List.concat (List.map (fun x_block ->
          List.map (fun y_block ->
            let x_mean = Tensor.mean x_block ~dim:[0] ~keepdim:false in
            let y_mean = Tensor.mean y_block ~dim:[0] ~keepdim:false in
            Tensor.(x_mean - y_mean)
          ) y_blocks
        ) x_blocks)
      in
      
      let med = median_block block_estimates in
      
      let w' = Tensor.(w + med * f learning_rate) in
      let w'' = clip_weights w' c in
      
      train_loop w'' (t + 1)
  in
  
  train_loop w 0