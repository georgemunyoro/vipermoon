function fib_iterative(n)
	local a, b = 0, 1
	if n < 1 then
		return 0
	end
	for i = 1, n do
		a, b = b, a + b
	end
	return a
end

for i = 0, 10 do
	local foo = fib_iterative(i)
	print(foo)
end
