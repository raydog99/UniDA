open Torch

type t = {
  generator : Module.t;
  critic : Module.t;
}

let create input_dim hidden_dim output_dim =
  let generator = 
    Sequential.(
      [ linear ~input_dim hidden_dim ~activation:Relu;
        linear ~input_dim:hidden_dim output_dim ])
  in
  let critic = 
    Sequential.(
      [ linear ~input_dim output_dim ~activation:Relu;
        linear ~input_dim:hidden_dim 1 ])
  in
  { generator; critic }

let train t data n_iter batch_size n_critic learning_rate =
  let optimizer_g = Optimizer.adam (Module.parameters t.generator) ~lr:learning_rate in
  let optimizer_c = Optimizer.adam (Module.parameters t.critic) ~lr:learning_rate in
  
  let rec train_loop t iter =
    if iter >= n_iter then t
    else
      let rec critic_loop i =
        if i >= n_critic then ()
        else
          let real_data = Tensor.narrow data ~dim:0 ~start:(iter * batch_size) ~length:batch_size in
          let noise = Tensor.randn [batch_size; Tensor.shape real_data |> List.tl |> List.hd] in
          let fake_data = Module.forward t.generator noise in
          
          let real_loss = Module.forward t.critic real_data |> Tensor.mean in
          let fake_loss = Module.forward t.critic fake_data |> Tensor.mean in
          let critic_loss = Tensor.(fake_loss - real_loss) in
          
          Optimizer.zero_grad optimizer_c;
          Tensor.backward critic_loss;
          Optimizer.step optimizer_c;
          
          critic_loop (i + 1)
      in
      
      critic_loop 0;
      
      let noise = Tensor.randn [batch_size; Tensor.shape data |> List.tl |> List.hd] in
      let fake_data = Module.forward t.generator noise in
      let generator_loss = Tensor.neg (Module.forward t.critic fake_data |> Tensor.mean) in
      
      Optimizer.zero_grad optimizer_g;
      Tensor.backward generator_loss;
      Optimizer.step optimizer_g;
      
      train_loop t (iter + 1)
  in
  
  train_loop t 0

let generate t n =
  let noise = Tensor.randn [n; Module.input_dim t.generator] in
  Module.forward t.generator noise