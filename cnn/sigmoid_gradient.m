function g = sigmoid_gradient(z)

	% Here z is already the output of a sigmoid
	g = z.*(1-z);

end
