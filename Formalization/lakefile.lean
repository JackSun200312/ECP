import Lake
open Lake DSL

-- Package definition
package «Formalization»

-- Declare your Lean library
lean_lib «utils» {}

-- Declare an executable that depends on mathlib
lean_exe «formalization» {
  root := `Main
}

-- Declare mathlib as a dependency
require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.19.0"
