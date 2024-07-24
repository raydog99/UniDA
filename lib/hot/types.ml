open Torch

type cluster = Tensor.t
type dataset = cluster array
type correspondence_matrix = Tensor.t

exception HiwaError of string