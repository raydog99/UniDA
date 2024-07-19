open Torch

module type S = sig
  val sinkhorn :
    ?iterations:int ->
    ?epsilon:float ->
    cost:Tensor.t ->
    a:Tensor.t ->
    b:Tensor.t ->
    unit ->
    Tensor.t * Tensor.t * Tensor.t
end

module Make () : S = struct
  let sinkhorn ?(iterations=100) ?(epsilon=1e-3) ~cost ~a ~b () =
    let k = Tensor.(neg (div_scalar cost epsilon) |> exp) in
    let u = Tensor.ones_like a in
    
    let rec loop t u =
      if t >= iterations then u
      else
        let v = Tensor.(div b (mm k (unsqueeze u (-1)))) in
        let u_new = Tensor.(div a (mm (transpose k ~dim0:(-2) ~dim1:(-1)) (unsqueeze v (-1)))) in
        loop (t + 1) u_new
    in
    
    let u_final = loop 0 u in
    let v_final = Tensor.(div b (mm k (unsqueeze u_final (-1)))) in
    let t_final = Tensor.(mul (mul (unsqueeze u_final (-1)) k) (unsqueeze v_final (-2))) in
    
    let alpha = Tensor.(neg (log u_final) |> mul_scalar epsilon) in
    let beta = Tensor.(neg (log v_final) |> mul_scalar epsilon) in
    
    (t_final, alpha, beta)
end