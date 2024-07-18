module type Config = sig
  val num_workers : int
  val num_points : int
  val step_size : float
  val max_iterations : int
end

module Make (C : Config) = struct
  module M = Master.Make(C)

  let run () =
    let sample_fns = List.init C.num_workers (fun _ ->
      fun () -> Torch.Tensor.rand [C.num_points]
    ) in
    M.run sample_fns
end