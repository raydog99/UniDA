open Base
open Data
open Torch
open Classifier

val adapt : Data.dataset -> Data.dataset -> float -> float -> Tensor.t option -> (Data.dataset, string) Result.t

val train_and_evaluate : Classifier.t -> Data.dataset -> Data.dataset -> (float, string) Result.t

val run : Data.dataset -> Data.dataset list -> float -> float -> (float list, string) Result.t