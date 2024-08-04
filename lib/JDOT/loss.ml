open Torch

type t =
    | MSE
    | CrossEntropy
    | SquaredHinge

let compute = function
    | MSE -> fun y_true y_pred -> Tensor.(mean (pow (sub y_true y_pred) (Scalar.f 2.)))
    | CrossEntropy -> fun y_true y_pred ->
        let softmax_output = Tensor.softmax y_pred ~dim:1 in
        Tensor.nll_loss y_true softmax_output
    | SquaredHinge -> fun y_true y_pred ->
        let margin = Tensor.(maximum (sub (f 1.) (mul y_true y_pred)) (f 0.)) in
        Tensor.(mean (pow margin (Scalar.f 2.)))

let to_string = function
    | MSE -> "MSE"
    | CrossEntropy -> "CrossEntropy"
    | SquaredHinge -> "SquaredHinge"

let of_string = function
    | "MSE" -> MSE
    | "CrossEntropy" -> CrossEntropy
    | "SquaredHinge" -> SquaredHinge
    | _ -> invalid_arg "Unknown loss function"