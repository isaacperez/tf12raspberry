Imagen:

   C1 ... CC
F1 RGB    RGB
.
.
.
FF RGB    RGB

  - En memoria: [(F1, C1) RGB, ..., (FF,CC) RGB] vector de tamaño FF*CC*num_canales(3 para RGB)

  - Cada píxel se codifica con un byte. Una fila de la imagen (step) tiene el tamaño: (CC*num_canales*Tam_de_un_byte)

  - Para acceder a la posición (fila_i, col_i, canal_i): fila_i*step + col_i*(num_canales*tam_de_un_byte) + canal_i*tam_de_un_byte

- Orden de compilación: gcc -I./headers/ -Wall ./sources/main.c ./sources/imagen.c ./sources/funciones.c -o ./bin/main
