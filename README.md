<div align="center">
  <h1>
    CHIMERA
  </h1>
</div>
The Chimera language may not be all that useful but it is definitely intersting.

Syntax is flexible and sometimes too wild.

# Features
In Chimera, the only way to produce output is through the ```print``` definition
```
print 1 + 2
```

# Questionable Features
You can make definitions with the ```define``` keyword.
Definitions are like functions, only more flexible syntax-wise.

Here is one way to define ```add```.
```
define add(a:int, b:int) -> int
  return a + b

print add(1, 2)
```
But the parenthesies are not necessary. And neither is the comma.
```
define add a:int b:int -> int
  return a + b

print add 1 2
```
Furthermore, the definition does not have to start with a name. And the name can be almost anything!

(including no name, but don't do that)
```
define a:int $ b:int -> int
  return a + b

print 1 $ 2
```
# Bad Features
Chimera obviously follows operator precedence.
```
print 1 + 2 * 3
=> 7
```
But it is not obvious that it has to.
```
precedence * = -1

print 1 + 2 * 3
=> 9
```
