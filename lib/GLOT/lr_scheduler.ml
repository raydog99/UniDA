module Optimizer = struct
  type t = {
    mutable param_groups: param_group list
  }
  and param_group = {
    mutable lr: float;
    mutable weight_decay: float;
    lr_mult: float;
    decay_mult: float;
  }

  let create param_groups =
    { param_groups }
end

let inv_lr_scheduler optimizer iter_num gamma power ?(lr = 0.001) ?(weight_decay = 0.0005) () =
  let lr = lr *. (1. +. gamma *. float_of_int iter_num) ** (-. power) in
  List.iter (fun param_group ->
    param_group.Optimizer.lr <- lr *. param_group.Optimizer.lr_mult;
    param_group.Optimizer.weight_decay <- weight_decay *. param_group.Optimizer.decay_mult;
  ) optimizer.Optimizer.param_groups;
  optimizer

module Schedule = struct
  type scheduler = Optimizer.t -> int -> float -> float -> ?lr:float -> ?weight_decay:float -> unit -> Optimizer.t

  let schedule_dict = [
    ("inv", inv_lr_scheduler)
  ]

  let get_scheduler name =
    List.assoc_opt name schedule_dict
end