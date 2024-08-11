open Base

type t

val create : int -> int -> (t, string) Result.t

val train : t -> Data.dataset -> (t, string) Result.t

val evaluate : t -> Data.dataset -> (float, string) Result.t