<div align="center">
  <h1>
    CHIMERA
  </h1>
</div>
The Chimera language may not be all that useful, but it is definitely interesting.

Syntax is flexible and mostly too wild.

# Features
In Chimera, the only way to produce output is through the ```print``` definition:

`
print 1 + 2
`

# Questionable Features
You can create definitions using the `define` keyword. Definitions are similar to functions but have more flexible syntax.

Here’s one way to define `add`:
```
define add(a:int, b:int) -> int
  return a + b

print add(1, 2)
```
However, parentheses are optional, and so is the comma:
```
define add a:int b:int -> int
  return a + b

print add 1 2
```
Furthermore, the definition doesn’t have to start with a name, and the name can be almost anything — even no name at all (but don’t do that):
```
define a:int $ b:int -> int
  return a + b

print 1 $ 2
```
# Bad Features
Chimera follows standard operator precedence:
```
print 1 + 2 * 3
=> 7
```
But it doesn’t have to:
```
precedence * = -1

print 1 + 2 * 3
=> 9
```
