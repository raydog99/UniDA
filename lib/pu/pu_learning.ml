open Torch

type dataset = Tensor.t

let create_pu_constraint p q s =
  let n, m = Tensor.shape p, Tensor.shape q in
  let constraint_matrix = Tensor.zeros [n+1; m+1] in
  Tensor.copy_ ~src:p ~dst:(Tensor.narrow constraint_matrix ~dim:0 ~start:0 ~length:n);
  Tensor.fill_ (Tensor.narrow constraint_matrix ~dim:1 ~start:m ~length:1) ((Tensor.sum p |> Tensor.item) -. s);
  Tensor.fill_ (Tensor.narrow constraint_matrix ~dim:0 ~start:n ~length:1) ((Tensor.sum q |> Tensor.item) -. s);
  constraint_matrix

let pu_wasserstein unlabeled positive pi alpha =
  let c = Utils.compute_euclidean_cost_matrix unlabeled positive in
  let p = Tensor.full [Tensor.shape2_exn unlabeled |> fst] (1. -. alpha) in
  let q = Tensor.full [Tensor.shape2_exn positive |> fst] (pi +. alpha) in
  PartialOT.partial_wasserstein c p q pi

let pu_gromov_wasserstein unlabeled positive pi alpha =
  let cs = Utils.compute_gromov_cost_matrix unlabeled in
  let ct = Utils.compute_gromov_cost_matrix positive in
  let p = Tensor.full [Tensor.shape2_exn unlabeled |> fst] (1. -. alpha) in
  let q = Tensor.full [Tensor.shape2_exn positive |> fst] (pi +. alpha) in
  PartialOT.partial_gromov_wasserstein cs ct p q pi

let classify transport_plan =
  let n, _ = Tensor.shape2_exn transport_plan in
  let row_sums = Tensor.sum transport_plan ~dim:[1] in
  Tensor.ge row_sums (Tensor.scalar_tensor 0.)