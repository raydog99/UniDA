open Torch

type t =
  | NegativeEntropy of float
  | SquaredNorm of float
  | GroupLasso of float * float * (int list) list

(** Apply negative entropy regularization *)
let negative_entropy gamma y =
  let eps = 1e-10 in
  let y_safe = Tensor.clamp_min y eps in
  Tensor.mul_scalar (Tensor.sum (Tensor.mul y_safe (Tensor.log y_safe))) gamma

(** Apply squared norm regularization *)
let squared_norm gamma y =
  Tensor.mul_scalar (Tensor.norm_sqr y) (gamma /. 2.)

(** Apply group lasso regularization *)
let group_lasso gamma mu groups y =
  let squared_term = squared_norm gamma y in
  let group_term = 
    List.fold_left (fun acc group ->
      let y_group = Tensor.index_select y 0 (Tensor.of_int1 group) in
      Tensor.add acc (Tensor.norm y_group)
    ) (Tensor.zeros []) groups
  in
  Tensor.add squared_term (Tensor.mul_scalar group_term (gamma *. mu))

(** Apply the specified regularization *)
let apply = function
  | NegativeEntropy gamma -> negative_entropy gamma
  | SquaredNorm gamma -> squared_norm gamma
  | GroupLasso (gamma, mu, groups) -> group_lasso gamma mu groups

(** Validate regularization parameters *)
let validate = function
  | NegativeEntropy gamma ->
      if gamma <= 0. then
        invalid_arg "Negative entropy regularization parameter must be positive"
  | SquaredNorm gamma ->
      if gamma <= 0. then
        invalid_arg "Squared norm regularization parameter must be positive"
  | GroupLasso (gamma, mu, groups) ->
      if gamma <= 0. || mu <= 0. then
        invalid_arg "Group lasso regularization parameters must be positive";
      if List.exists (fun g -> g = []) groups then
        invalid_arg "Group lasso groups must not be empty"