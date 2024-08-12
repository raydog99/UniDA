open Torch

type t = Optimizer.t

let adam = Optimizer.adam
let sgd = Optimizer.sgd
let zero_grad = Optimizer.zero_grad
let step = Optimizer.step