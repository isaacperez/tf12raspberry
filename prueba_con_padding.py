print("Esquina superior izquierda de la imagen")
kernel = range(1,10)
input = [15.0, 15.0,19.0,15.0,15.0,19.0,39.0,39.0,43.0]
c0 = 0
for i in range(0,9):
    c0 += float(kernel[i])*input[i]

print("Filtro 0, Canal 0", c0)

kernel = range(10,19)
input = [i+1 if i!=0 else i for i in input]
c1 = 0
for i in range(0,9):
    c1 += float(kernel[i])*input[i]

print("Filtro 0, Canal 1", c1)

print("Resultado del filtro 0:", c0+c1)


kernel = range(19,28)
input = [15.0, 15.0,19.0,15.0,15.0,19.0,39.0,39.0,43.0]
c0 = 0
for i in range(0,9):
    c0 += float(kernel[i])*input[i]

print("Filtro 1, Canal 0", c0)

kernel = range(28,37)
input = [i+1 if i!=0 else i for i in input]
c1 = 0
for i in range(0,9):
    c1 += float(kernel[i])*input[i]

print("Filtro 1, Canal 1", c1)

print("Resultado del filtro 1:", c0+c1)


print("\nCentro de la imagen")
kernel = range(1,10)
input = [15, 19, 23, 39, 43, 47, 63, 67, 71]
c0 = 0
for i in range(0,9):
    c0 += float(kernel[i])*input[i]

print("Filtro 0, Canal 0", c0)

kernel = range(10,19)
input = [i+1 if i!=0 else i for i in input]
c1 = 0
for i in range(0,9):
    c1 += float(kernel[i])*input[i]

print("Filtro 0, Canal 1", c1)

print("Resultado del filtro 0:", c0+c1)

kernel = range(19,28)
input = [15, 19, 23, 39, 43, 47, 63, 67, 71]
c0 = 0
for i in range(0,9):
    c0 += float(kernel[i])*input[i]

print("Filtro 1, Canal 0", c0)

kernel = range(28,37)
input = [i+1 if i!=0 else i for i in input]
c1 = 0
for i in range(0,9):
    c1 += float(kernel[i])*input[i]

print("Filtro 1, Canal 1", c1)

print("Resultado del filtro 1:", c0+c1)


print("\nEsquina inferior derecha")
kernel = range(1,10)
input = [43, 47, 47, 67, 71, 71, 67, 71, 71]
c0 = 0
for i in range(0,9):
    c0 += float(kernel[i])*input[i]

print("Filtro 0, Canal 0", c0)

kernel = range(10,19)
input = [i+1 if i!=0 else i for i in input]
c1 = 0
for i in range(0,9):
    c1 += float(kernel[i])*input[i]

print("Filtro 0, Canal 1", c1)

print("Resultado del filtro 0:", c0+c1)

kernel = range(19,28)
input = [43, 47, 47, 67, 71, 71, 67, 71, 71]
c0 = 0
for i in range(0,9):
    c0 += float(kernel[i])*input[i]

print("Filtro 1, Canal 0", c0)

kernel = range(28,37)
input = [i+1 if i!=0 else i for i in input]
c1 = 0
for i in range(0,9):
    c1 += float(kernel[i])*input[i]

print("Filtro 1, Canal 1", c1)

print("Resultado del filtro 1:", c0+c1)
