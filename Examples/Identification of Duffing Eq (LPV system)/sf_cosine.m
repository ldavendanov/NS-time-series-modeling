function y = sf_cosine(t,f0,t1,f1)

if t <= t1
    beta = (f1-f0)/(2*t1);
    f = f0 + beta*t;
else
    beta = (f1-f0)/(2*t1);
    f = 2*f1 - beta*t;
end

y = cos(2*pi*f.*t);