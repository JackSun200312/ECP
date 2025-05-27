import Lean
import Mathlib
import Aesop
open Lean Elab Tactic Meta Command Expr

syntax "get_state" : tactic
syntax "run_coder" : tactic
syntax "run_conjecturer" : tactic
syntax "print_exists_type" : tactic
syntax (name := printFol) "print_fol" : tactic
syntax "try_solvers" : tactic


def getPpTacticState : TacticM String := do
  let goals ← getUnsolvedGoals
  match goals with
  | [] => return "no goals"
  | [g] => return (← Meta.ppGoal g).pretty
  | gs =>
    return (← gs.foldlM (init := "") (fun acc g => do
      return acc ++ "\n\n" ++ (← Meta.ppGoal g).pretty)).trim

def getExistsTypesLoop (e : Expr) : MetaM (List (Name × Expr)) := do
  let mut result : List (Name × Expr) := []
  let mut current := e

  while true do
    let currentWhnf ← whnf current
    match currentWhnf with
    | Expr.app (Expr.app (Expr.const `Exists _) ty) lamExpr =>
      let lamExprWhnf ← whnf lamExpr
      match lamExprWhnf with
      | Expr.lam binderName binderType body _ =>
        result := result.append [(binderName, ty)]
        -- Use dummy variable to instantiate and move deeper
        let dummy := mkFVar ⟨Name.anonymous⟩
        current := body.instantiate1 dummy
      | _ => break
    | _ => break

  return result
def getExistsType : TacticM String := do
  let target ← getMainTarget
  let vars ← getExistsTypesLoop target
  let formatted ← vars.mapM fun (n, ty) => do
    let tyStr ← ppExpr ty
    return s!"{n.toString} : {tyStr.pretty}"
  return (formatted.toString.replace "[" "").replace "]" ""


def runECPWithState (function : String) (state : String): TacticM String := do
  let child ← IO.Process.spawn {
    cmd := "python",
    args := #["src/scripts/lean/run_ecp.py" , "--state", state, "--function", function],
    stdin := .piped,
    stdout := .piped,
    stderr := .piped,
    cwd := "/fs01/projects/imosolver/ECP/"
  }
  child.stdin.flush

  let output ← child.stdout.readToEnd
  -- let err ← child.stderr.readToEnd
  -- if err.trim ≠ "" then
  --   logInfo m!"[stderr from {function}]\n{err.trim}"
  return output.trim


private def collectHypsAndGoal (type : Expr) : MetaM (Array (Name × Expr) × List Expr × Expr) := do
  let typeWhnf ← whnf type
  logInfo type
  logInfo typeWhnf
  forallTelescope typeWhnf fun xs body => do
    logInfo xs
    let mut vars := #[]
    let mut hyps := []

    for x in xs do
      let xType ← inferType x
      if (← isProp xType) then
        hyps := hyps ++ [xType]
      else
        let localDecl ← getFVarLocalDecl x
        vars := vars.push (localDecl.userName, xType)

    return (vars, hyps, body)



elab_rules : tactic
  | `(tactic| get_state) => do
    let state ← getPpTacticState
    let filename ← getFileName
    let content ← IO.FS.readFile filename

    logInfo m!"State: {state}, Filename: {filename}, File content: {content}"


  | `(tactic| run_coder) => do
    let state ← getPpTacticState
    let result ←  runECPWithState "coder" state
    logInfo m!"[enumerate_solution]\n{result}"

  | `(tactic| run_conjecturer) => do
    let state ← getPpTacticState
    let result ←  runECPWithState "conjecturer" state
    logInfo m!"[find_solution]\n{result}"

  | `(tactic| print_exists_type) => do
    let vars ← getExistsType
    logInfo m!"{vars}"


  | `(tactic| print_fol) => do
    let _    ← getMainGoal
    let lctx ← getLCtx

    -- Partition: non-Prop locals as vars, Prop-typed locals by storing their .type
    let mut vars  : Array (Name × Expr) := #[]
    let mut props : Array Expr          := #[]
    for decl in lctx do
      if !decl.isAuxDecl then
        if ← isProp decl.type then
          props := props.push decl.type    -- <<< store the proposition itself
        else
          vars  := vars.push (decl.userName, decl.type)

    -- Group same-typed vars into (names, ty) pairs
    let mut groups : Array (Array Name × Expr) := #[]
    for (n, ty) in vars do
      if groups.isEmpty then
        groups := groups.push (#[n], ty)
      else
        let (ns, ty₀) := groups.get! (groups.size - 1)
        if ← isDefEq ty₀ ty then
          groups := groups.set! (groups.size - 1) (ns.push n, ty)
        else
          groups := groups.push (#[n], ty)

    -- Build the ∀-prefix
    let mut quantStr : String := ""
    for (ns, ty) in groups do
      let tyDoc ← ppExpr ty
      let names := ns.toList.map Name.toString
      quantStr := quantStr ++ s!"∀ {String.intercalate " " names} : {tyDoc.pretty}, "

    -- Build the conjunction of props (now the actual types)
    let propsStr : String ←
      if props.isEmpty then
        pure ""
      else do
        let docStrs ← props.mapM fun p => do
          let d ← ppExpr p
          pure d.pretty
        let ss := docStrs.toList
        pure s!" {String.intercalate " ∧ " ss} → "

    -- Finally, get and pretty-print the main goal
    let tgt    ← getMainTarget
    let tgtDoc ← ppExpr tgt

    logInfo m!"{quantStr}{propsStr}{tgtDoc.pretty}"

elab_rules : tactic
  | `(tactic| try_solvers) => do
    -- build an array of the tactics we want to try
    let tactics ← pure #[
      ← `(tactic| simp),
      ← `(tactic| aesop),
      ← `(tactic| nlinarith),
      ← `(tactic| ring),
      ← `(tactic| norm_num)
    ]

    -- get the initial list of goals
    let gsBefore ← getGoals
    let mut solved := false

    for tac in tactics do
      if ¬solved then
        let snapshot ← saveState
        try
          -- apply the candidate tactic
          evalTactic tac
          -- check how many goals remain
          let gsAfter ← getGoals
          if gsAfter.length < gsBefore.length then
            solved := true
          else
            -- it did nothing, roll back
            snapshot.restore
        catch _ =>
          -- it outright failed, roll back
          snapshot.restore

    if solved then
      logInfo m!"True"
    else
      logInfo m!"False"
