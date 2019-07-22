function F = NIDexcitation(t,f,fs)

k = round(t*fs);
F = f(k);