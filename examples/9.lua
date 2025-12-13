local x = { 1, 2, 3, x = 10 }
local y = { 1, 2, 3 }
local z = { foo = { 42 } }
print(x.x, y[1], z.foo[1])
