chimera is programming language for accelerated computing
with this goal in mind, a few restrictions are in order:

- No unbounded loops or recursion.
  This restriction makes the language non-turing complete and thus restricts the expressiveness of the language.
  chimera is supposed to be used in tandem with another programming language in order to achieve expressiveness.
  With this restriction a few optimizations can be achieved
  - Programs are guaranteed to finish
    With turing completenes gone, the halting problem is also gone
  - Compute budget estimation
    The compute of a program can be estimated as a range. It cannot be estimated exactly because of conditionals.
- Functional paradigm
  This may seem to contradict the notion of accelerated computing and it may also in fact do that.
  I like functional paradigm so I will try to make it work
  Also functions can be parallelized as they are pure, which may be harder to infer from loops
  I want implicit parallelization as default and allow customization if needed
  Since functions are pure, I can cache results for the same input parameters.

---
As noted earlier, chimera is supposed to be used in tandem with another programming language and rarely on its own.
Therefore the language as no print or input functions. chimera takes input arguments when it is called and outputs a value when it completes.
There is no outside influnce on the program while it is running.

chimera programs are run top to bottom

Example chimera program
```
a = 3
b = 5
a + b
```
The program will return 8 and exit

Though chimera does not have a print function, it does have `dbg`, which will print to stdout and can be used during development.

In chimera scalars and arrays can be expressed interchangably
Take the following function:
```
def step_simulation(x, v, a, dt)
  v += a * dt
  x += v * dt
  x, v
```
The function can either be called with scalars or arrays
```
step_simulation(0, 10, -9.81, 0.02)
step_simulation([1, 2, 3], [10, 20, 30], [0, -9.81, 0], 0.02)
```

Most operations on arrays are elementwise and arrays of different shapes will attempt to broadcast

List of compilation steps
- Tokenize
- Parse
  - Syntax validation
- Rewrite
  - Structural rewrite
  - Index lowering
  - Symbolic
- Validation
  - Type checking
  - Shape checking
  - Semantic analysis
- Linearize
- Render
- Compile