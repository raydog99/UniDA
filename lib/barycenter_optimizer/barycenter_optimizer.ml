open Torch

module type Config = sig
  val epsilon : float
  val beta : float
  val alpha : float
  val t_max : int
  val j_max : int
  val n_samples : int
  val n_features : int
end

module Make (C : Config) = struct
  type t = {
    mutable phi: Tensor.t;
  }

  let create initial_phi =
    { phi = initial_phi }

  let compute_cost_matrix () =
    let x = Tensor.arange ~end_:(Float.of_int C.n_features) ~options:(Kind Float, Device.Cpu) in
    let x = Tensor.reshape x [-1; 1] in
    let cost_matrix = Tensor.(pow_scalar (sub x (transpose_dim x ~dim0:(-2) ~dim1:(-1))) 2.) in
    cost_matrix

  let sinkhorn ~cost_matrix ~mu ~nu ~n_iter =
    let k = Tensor.exp Tensor.(div (neg cost_matrix) C.epsilon) in
    let u = Tensor.ones_like mu in
    let rec loop i u =
      if i = 0 then u
      else
        let v = Tensor.div nu Tensor.(matmul k (reshape u [-1; 1])) in
        let u' = Tensor.div mu Tensor.(matmul (transpose k) (reshape v [-1; 1])) in
        loop (i - 1) u'
    in
    let u = loop n_iter u in
    let v = Tensor.div nu Tensor.(matmul k (reshape u [-1; 1])) in
    let pi = Tensor.(mul k (mul (reshape u [-1; 1]) (reshape v [1; -1]))) in
    pi

  let compute_gradient cost_matrix pi mu nu =
    let n = Tensor.size mu 0 in
    let grad = Tensor.(sum (mul cost_matrix pi) ~dim:[1]) in
    Tensor.(reshape grad [n; 1])

  let compute_a_hat t data =
    let cost_matrix = compute_cost_matrix () in
    let mu = Tensor.ones [C.n_samples] ~kind:Float in
    let nu = t.phi in
    let pi = sinkhorn ~cost_matrix ~mu ~nu ~n_iter:100 in
    compute_gradient cost_matrix pi mu nu

  let compute_b_hat t data =
    let cost_matrix = compute_cost_matrix () in
    let mu = t.phi in
    let nu = Tensor.ones [C.n_samples] ~kind:Float in
    let pi = sinkhorn ~cost_matrix ~mu ~nu ~n_iter:100 in
    compute_gradient cost_matrix pi mu nu

  let optimize t data =
    for t_iter = 1 to C.t_max do
      for j = 1 to C.j_max do
        let z = ref (Tensor.zeros_like t.phi) in
        let w = ref t.phi in
        
        let rec gradient_descent () =
          let grad = compute_a_hat t data in
          if Tensor.(norm grad) > C.epsilon then begin
            z := Tensor.(C.beta * !z + grad);
            w := Tensor.(!w - C.alpha * !z);
            gradient_descent ()
          end
        in
        gradient_descent ();

        t.phi <- !w;
      done;

      let b_hat = compute_b_hat t data in
      t.phi <- Tensor.(div_scalar (add t.phi b_hat) 2.);
    done;

    t.phi
end