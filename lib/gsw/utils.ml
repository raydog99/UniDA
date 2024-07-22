open Torch

let sort_tensor tensor =
  let sorted, _ = Tensor.sort tensor ~dim:0 ~descending:false in
  sorted

let l_p_distance p x y =
  Tensor.(pow (abs (sub x y)) (of_float p) |> sum ~dim:[0] |> pow (of_float (1. /. p)))

let empirical_pdf samples =
  let n = Tensor.shape samples |> List.hd |> float_of_int in
  Tensor.of_float 1. /. Tensor.of_float n

let unpack_nn_params theta input_dim hidden_dim num_layers =
  let total_params = input_dim * hidden_dim + hidden_dim * hidden_dim * (num_layers - 1) + hidden_dim * num_layers in
  assert (Tensor.shape theta = [total_params]);
  
  let rec build_layers acc offset layer =
    if layer >= num_layers then List.rev acc
    else
      let in_dim = if layer = 0 then input_dim else hidden_dim in
      let out_dim = if layer = num_layers - 1 then 1 else hidden_dim in
      let w_size = in_dim * out_dim in
      let b_size = out_dim in
      
      let w = Tensor.narrow theta 0 offset w_size |> Tensor.reshape [in_dim; out_dim] in
      let b = Tensor.narrow theta 0 (offset + w_size) b_size |> Tensor.reshape [1; out_dim] in
      
      build_layers ((w, b) :: acc) (offset + w_size + b_size) (layer + 1)
  in
  
  let layers = build_layers [] 0 0 in
  let weights, biases = List.split layers in
  weights, biases

let linear x theta =
  Tensor.mm x (Tensor.unsqueeze theta 1)

let circular x theta =
  let r = Tensor.of_float 1.0 in
  let centered = Tensor.sub x (Tensor.mul r theta) in
  Tensor.norm centered ~dim:[1] ~p:2 ~keepdim:false

let polynomial x theta ~degree =
  let batch_size = Tensor.shape x |> List.hd in
  let result = Tensor.zeros [batch_size] in
  
  for d = 0 to degree do
    let term = Tensor.pow (Tensor.mm x (Tensor.unsqueeze theta 1)) (Tensor.of_int d) in
    let coeff = Tensor.select theta 0 d in
    Tensor.add_ result (Tensor.mul term coeff)
  done;
  
  result

let neural_network x theta ~num_layers ~hidden_dim =
  let input_dim = Tensor.shape x |> List.tl |> List.hd in
  let weights, biases = Utils.unpack_nn_params theta input_dim hidden_dim num_layers in
  
  let rec forward input layer =
    if layer >= num_layers then input
    else
      let w = List.nth weights layer in
      let b = List.nth biases layer in
      let output = Tensor.(mm input w + b) in
      let activated = if layer = num_layers - 1 then output else Tensor.leaky_relu output in
      forward activated (layer + 1)
  in
  
  forward x 0