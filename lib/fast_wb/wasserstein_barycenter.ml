open Torch
open Utils

module type WassersteinBarycenter = sig
  val compute_barycenter :
    ?iterations:int ->
    ?t0:float ->
    ?lambda:float ->
    supports:Tensor.t list ->
    weights:Tensor.t list ->
    p:float ->
    init_support:Tensor.t ->
    theta:Tensor.t ->
    unit ->
    Tensor.t

  val proximal_update :
    a:Tensor.t -> b:Tensor.t -> t0:float -> alpha:Tensor.t -> Tensor.t
end

module Make (S : Sinkhorn.S) : WassersteinBarycenter = struct
  let proximal_update ~a ~b ~t0 ~alpha =
    let exp_term = Tensor.(neg (mul_scalar alpha t0) |> exp) in
    let updated = Tensor.(mul a exp_term) in
    kl_projection updated

  let compute_barycenter ?(iterations=100) ?(t0=1.0) ?(lambda=1.0)
      ~supports ~weights ~p ~init_support ~theta () =
    let n = List.length supports in
    let dim = Tensor.shape init_support in
    
    let rec loop t a x =
      if t >= iterations then x
      else
        let beta = (float t +. 1.) /. 2. in
        let alphas = 
          List.map2 (fun y w ->
            let cost = pairwise_distances x y p in
            let _, alpha, _ = S.sinkhorn ~epsilon:lambda ~cost ~a ~b:w () in
            alpha
          ) supports weights
        in
        let alpha_sum = 
          List.fold_left Tensor.add (Tensor.zeros_like (List.hd alphas)) alphas
          |> Tensor.div_scalar (float n)
        in
        let a_new = proximal_update ~a ~b:theta ~t0:(t0 *. beta) ~alpha:alpha_sum in
        let x_new = 
          List.map2 (fun y t_opt ->
            Tensor.(mm (transpose y ~dim0:(-2) ~dim1:(-1)) t_opt)
          ) supports alphas
          |> List.fold_left Tensor.add (Tensor.zeros_like (List.hd supports))
          |> Tensor.div_scalar (float n)
          |> Tensor.mm (Tensor.diag (Tensor.reciprocal a_new))
        in
        let x_interpolated = Tensor.(
          add (mul_scalar x (1. -. (1. /. beta))) (mul_scalar x_new (1. /. beta))
        ) in
        loop (t + 1) a_new x_interpolated
    in
    
    let a_init = Tensor.ones dim in
    loop 0 a_init init_support
end