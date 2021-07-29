input = [15.00, 16.00, 19.00, 20.00, 23.00, 24.00, 39.00, 40.00, 43.00, 44.00, 47.00, 48.00, 63.00, 64.00, 67.00, 68.00, 71.00, 72.00]
neurona1 = range(1,36, 2)
neurona2 = range(2,38, 2)

out1 = 0
for i in range(18):
    out1 += float(neurona1[i]) * input[i]

out2 = 0
for i in range(18):
    print(neurona2[i],"*", input[i])
    out2 += float(neurona2[i]) * input[i]

print(out1, out2)
