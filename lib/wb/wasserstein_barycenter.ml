open Torch
open Types
open Utils

type t = {
  measures : measure array;
  cost_matrix : cost_matrix;
  incidence_matrix : incidence_matrix;
  step_size : float;
  num_iterations : int;
}

let create measures cost_matrix incidence_matrix epsilon =
  let m = Array.length measures in
  let n = Tensor.shape measures.(0) |> List.hd in
  let step_size = 1. /. (4. *. Tensor.l1_norm cost_matrix *. sqrt (6. *. float_of_int n *. log (float_of_int n))) in
  let num_iterations = int_of_float (8. *. Tensor.l1_norm cost_matrix *. sqrt (6. *. float_of_int n *. log (float_of_int n)) /. epsilon) in
  { measures; cost_matrix; incidence_matrix; step_size; num_iterations }

let run t =
  let m = Array.length t.measures in
  let n = Tensor.shape t.measures.(0) |> List.hd in
  let n2 = n * n in

  let alpha = 2. *. Tensor.l1_norm t.cost_matrix *. t.step_size *. float_of_int n in
  let beta = 6. *. Tensor.l1_norm t.cost_matrix *. t.step_size *. log (float_of_int n) in
  let gamma = 3. *. float_of_int m *. t.step_size *. log (float_of_int n) in

  let p = Tensor.full [n] (1. /. float_of_int n) in
  let x = Array.init m (fun _ -> Tensor.full [n2] (1. /. float_of_int n2)) in
  let y = Array.init m (fun _ -> Tensor.zeros [2 * n]) in

  let rec iterate k p x y =
    if k >= t.num_iterations then (p, x, y)
    else
      let v = Array.mapi (fun i yi ->
        let vi = Tensor.(yi + alpha * (matmul t.incidence_matrix (x.(i) - (p / t.measures.(i))))) in
        project_onto_cube vi
      ) y in

      let u = Array.mapi (fun i xi ->
        let ui = Tensor.(xi * exp (-(t.cost_matrix + 2. * l1_norm t.cost_matrix * transpose t.incidence_matrix @@ y.(i)) / (float_of_int n2))) in
        project_onto_simplex ui
      ) x in

      let s = Tensor.(p * exp (-(sum (Array.map (fun yi -> yi.(!![0 -- (n-1)])) v) ~dim:[0]) / float_of_int n)) in
      let s = project_onto_simplex s in

      let y' = Array.mapi (fun i vi ->
        let yi' = Tensor.(vi + alpha * (matmul t.incidence_matrix (u.(i) - (s / t.measures.(i))))) in
        project_onto_cube yi'
      ) v in

      let x' = Array.mapi (fun i xi ->
        let xi' = Tensor.(xi * exp (-(t.cost_matrix + 2. * l1_norm t.cost_matrix * transpose t.incidence_matrix @@ v.(i)) / (float_of_int n2))) in
        project_onto_simplex xi'
      ) x in

      let p' = Tensor.(p * exp (-(sum (Array.map (fun vi -> vi.(!![0 -- (n-1)])) v) ~dim:[0]) / float_of_int n)) in
      let p' = project_onto_simplex p' in

      iterate (k + 1) p' x' y'
  in

  iterate 0 p x y