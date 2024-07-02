open Torch

module WassersteinDistance = struct
  let compute_c x y =
    Tensor.mse_loss x y Reduction.None

  let compute_a c beta =
    Tensor.exp (Tensor.div c (Scalar.f beta))

  let hadamard_product a b =
    Tensor.mul a b

  let frobenius_dot_product a b =
    Tensor.sum (Tensor.mul a b)

  let compute_wasserstein_distance x y beta max_iter k =
    let n = Tensor.size x 0 in
    let sigma = Tensor.full [n] (1.0 /. float_of_int n) in
    let t = Tensor.ones [n; n] in
    
    let c = compute_c x y in
    let a = compute_a c beta in
    
    let rec iterate t iter =
      if iter > max_iter then t
      else
        let q = hadamard_product a t in
        
        let rec sinkhorn_step q k_iter =
          if k_iter > k then q
          else
            let delta = Tensor.div (Scalar.f 1.0) (Tensor.mv q sigma) in
            let new_sigma = Tensor.div (Scalar.f 1.0) (Tensor.mv (Tensor.t q) delta) in
            sinkhorn_step q (k_iter + 1)
        in
        
        let q_final = sinkhorn_step q 1 in
        
        let delta = Tensor.div (Scalar.f 1.0) (Tensor.mv q_final sigma) in
        let new_t = Tensor.mul (Tensor.diag delta) (Tensor.mul q_final (Tensor.diag sigma)) in
        
        iterate new_t (iter + 1)
    in
    
    let final_t = iterate t 1 in
    let d_wd = frobenius_dot_product (Tensor.t c) final_t in
    
    (final_t, d_wd)

  let run x y beta max_iter k =
    let t, d_wd = compute_wasserstein_distance x y beta max_iter k in
    (t, d_wd)
end