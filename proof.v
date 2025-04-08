(* Import necessary libraries *)
Require Import Reals Lra. (* Standard library for Real numbers and Linear Real Arithmetic *)
Require Import Coquelicot.Coquelicot. (* The Coquelicot library for real analysis *)

(* Open scopes for convenient notation *)
Open Scope R_scope.

(* --- Verification of Gradient Calculation --- *)

(* This file verifies the partial derivatives of the average batch loss function
   used in the Python test `test_bgd_symbolic_verification`.

   The loss function is L(m, b) = ( |m*x₁ + b - y₁| + |m*x₂ + b - y₂| ) / 2
   We want to compute ∂L/∂m and ∂L/∂b at the point (m₀, b₀).
*)

(* Define the constants used in the test case as Real numbers (R) *)
Definition x1 : R := 2.
Definition y1 : R := 2.
Definition x2 : R := 3.
Definition y2 : R := 3.

(* Define the point (m₀, b₀) at which we evaluate the derivatives *)
Definition m0 : R := 20.
Definition b0 : R := 10.

(* Define the average batch loss function L(m, b) *)
Definition L (m b : R) : R :=
  (Rabs (m * x1 + b - y1) + Rabs (m * x2 + b - y2)) / 2.

(* --- Computing Partial Derivatives --- *)

(* We compute the partial derivatives by treating the function as a single-variable
   function while holding the other variable constant, then using `Derive` from Coquelicot.

   * ∂L/∂m at (m₀, b₀) is `Derive (fun m => L m b0) m0`
   * ∂L/∂b at (m₀, b₀) is `Derive (fun b => L m0 b) b0`

   We use `Lemma` to state the goal. Coq's tactics like `lra` (Linear Real Arithmetic)
   and Coquelicot's `auto with real_derive` will compute the result after simplification.
*)

(* Helper lemma: Prove the argument inside the first Rabs is positive at (m0, b0) *)
Lemma arg1_pos : 0 < m0 * x1 + b0 - y1.
Proof.
  unfold m0, x1, b0, y1. (* Replace definitions with values *)
  lra. (* Linear Real Arithmetic tactic solves the goal *)
Qed.

(* Helper lemma: Prove the argument inside the second Rabs is positive at (m0, b0) *)
Lemma arg2_pos : 0 < m0 * x2 + b0 - y2.
Proof.
  unfold m0, x2, b0, y2.
  lra.
Qed.

(* Compute ∂L/∂m at (m₀, b₀) and check if it equals 2.5 (5/2) *)
Lemma dLdm_at_m0b0 : Derive (fun m => L m b0) m0 = 5 / 2.
Proof.
  (* Unfold the definition of L and the constants in the point *)
  unfold L, b0.
  (* Apply derivative rules. `auto with real_derive` uses Coquelicot's database. *)
  (* We need the positivity facts proven above. *)
  auto with real_derive.
  (* At this point, the goal should be simplified symbolically. *)
  (* We need to show Rabs simplifies correctly because args are positive *)
  replace (Rabs (m0 * x1 + b0 - y1)) with (m0 * x1 + b0 - y1) by (apply Rabs_pos_eq; apply arg1_pos).
  replace (Rabs (m0 * x2 + b0 - y2)) with (m0 * x2 + b0 - y2) by (apply Rabs_pos_eq; apply arg2_pos).
  (* Unfold remaining constants and compute the final numerical result *)
  unfold x1, x2, y1, y2, m0, b0.
  field. (* Simplifies field expressions (like rational arithmetic) *)
  lra. (* Solves the final numerical equality *)
Qed.

(* Compute ∂L/∂b at (m₀, b₀) and check if it equals 1.0 (1/1) *)
Lemma dLdb_at_m0b0 : Derive (fun b => L m0 b) b0 = 1.
Proof.
  unfold L, m0.
  auto with real_derive.
  replace (Rabs (m0 * x1 + b0 - y1)) with (m0 * x1 + b0 - y1) by (apply Rabs_pos_eq; apply arg1_pos).
  replace (Rabs (m0 * x2 + b0 - y2)) with (m0 * x2 + b0 - y2) by (apply Rabs_pos_eq; apply arg2_pos).
  unfold x1, x2, y1, y2, m0, b0.
  field.
  lra.
Qed.
