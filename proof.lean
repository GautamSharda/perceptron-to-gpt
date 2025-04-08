-- Import necessary libraries from Mathlib
import Mathlib.Analysis.Calculus.Deriv.Basic -- For deriv (derivative of single-variable functions)
import Mathlib.Data.Real.Basic          -- For Real numbers and basic operations
import Mathlib.Tactic.NormNum           -- For numerical computation tactic

/-!
# Verification of Gradient Calculation

This file verifies the partial derivatives of the average batch loss function
used in the Python test `test_bgd_symbolic_verification`.

The loss function is L(m, b) = ( |m*x₁ + b - y₁| + |m*x₂ + b - y₂| ) / 2
We want to compute ∂L/∂m and ∂L/∂b at the point (m₀, b₀).
-/

-- Define the constants used in the test case as Real numbers
def x₁ : Real := 2
def y₁ : Real := 2
def x₂ : Real := 3
def y₂ : Real := 3

-- Define the point (m₀, b₀) at which we evaluate the derivatives
def m₀ : Real := 20
def b₀ : Real := 10

-- Define the average batch loss function L(m, b)
-- Takes two Real numbers (m, b) and returns a Real number
def L (m b : Real) : Real :=
  (abs (m * x₁ + b - y₁) + abs (m * x₂ + b - y₂)) / 2

/-!
## Computing Partial Derivatives

We compute the partial derivatives by treating the function as a single-variable
function while holding the other variable constant, then using `deriv`.

* ∂L/∂m at (m₀, b₀) is `deriv (fun m => L m b₀) m₀`
* ∂L/∂b at (m₀, b₀) is `deriv (fun b => L m₀ b) b₀`

We use `example` which is like a theorem without a name.
Lean's `norm_num` tactic will compute the numerical result after simplification.
Note: We use rational numbers like `(5 / 2 : Real)` for precision.
-/

-- Compute ∂L/∂m at (m₀, b₀) and check if it equals 2.5
example : deriv (fun m => L m b₀) m₀ = (5 / 2 : Real) := by
  -- Unfold the definitions to expose the expression
  unfold L x₁ x₂ y₁ y₂ b₀ m₀
  -- Check that the arguments to abs are non-zero at the point (m₀, b₀)
  -- m₀ * x₁ + b₀ - y₁ = 20 * 2 + 10 - 2 = 48 > 0
  -- m₀ * x₂ + b₀ - y₂ = 20 * 3 + 10 - 3 = 67 > 0
  -- Since they are positive, abs is differentiable and equals the identity
  -- Apply derivative rules and simplify the expression
  simp only [deriv_add, deriv_div_const, deriv_abs, deriv_add_const, deriv_sub_const,
             deriv_mul_const_field, deriv_id' (x := m₀), abs_pos]
  -- Compute the final numerical result
  norm_num

-- Compute ∂L/∂b at (m₀, b₀) and check if it equals 1.0
example : deriv (fun b => L m₀ b) b₀ = (1 : Real) := by
  -- Unfold the definitions
  unfold L x₁ x₂ y₁ y₂ b₀ m₀
  -- Arguments to abs are positive (checked above)
  -- Apply derivative rules and simplify
  simp only [deriv_add, deriv_div_const, deriv_abs, deriv_add_const, deriv_sub_const,
             deriv_const_mul_field, deriv_id' (x := b₀), abs_pos]
  -- Compute the final numerical result
  norm_num

-- Optional: Explicit proofs that the arguments to abs are positive at (m₀, b₀)
-- This confirms deriv_abs simplifies correctly via abs_pos in the simp call.
lemma arg1_pos : m₀ * x₁ + b₀ - y₁ > 0 := by norm_num
lemma arg2_pos : m₀ * x₂ + b₀ - y₂ > 0 := by norm_num
