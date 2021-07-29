BIN_DIR = ./bin/
HEADERS_DIR = ./headers/
SOURCE_DIR = ./sources/
OBJ_DIR = ./obj/
IMG_DIR = ./tensorflow/imagenes_de_test/
PARAM_DIR = ./tensorflow/ficheros_c/modelo_entrenado_mnist_cnn13/

all:
	gcc -g -I$(HEADERS_DIR) -Wall -O3 -c $(SOURCE_DIR)funciones.c -o $(OBJ_DIR)funciones.o
	#gcc -g-I$(HEADERS_DIR) -Wall -O3 -c $(SOURCE_DIR)main.c -o $(OBJ_DIR)main.o
	#gcc -g -I$(HEADERS_DIR) -I$(IMG_DIR) -I$(PARAM_DIR) -Wall -O3 -c $(SOURCE_DIR)main_mnist_fc.c -o $(OBJ_DIR)main_mnist_fc.o
	#gcc -g -I$(HEADERS_DIR) -I$(IMG_DIR) -I$(PARAM_DIR) -Wall -O3 -c $(SOURCE_DIR)main_mnist_cnn1.c -o $(OBJ_DIR)main_mnist_cnn1.o
	#gcc -g -I$(HEADERS_DIR) -I$(IMG_DIR) -I$(PARAM_DIR) -Wall -O3 -c $(SOURCE_DIR)main_mnist_cnn2.c -o $(OBJ_DIR)main_mnist_cnn2.o
	gcc -g -I$(HEADERS_DIR) -I$(IMG_DIR) -I$(PARAM_DIR) -Wall -O3 -c $(SOURCE_DIR)main_mnist_cnn13.c -o $(OBJ_DIR)main_mnist_cnn13.o
	gcc -g -I$(HEADERS_DIR) -Wall -O3 -c $(SOURCE_DIR)imagen.c -o $(OBJ_DIR)imagen.o

	#gcc -g-I$(HEADERS_DIR) -Wall -O3 $(OBJ_DIR)imagen.o $(OBJ_DIR)funciones.o $(OBJ_DIR)main.o -o $(BIN_DIR)main
	#gcc -g -I$(HEADERS_DIR) -I$(IMG_DIR) -I$(PARAM_DIR) -Wall -O3 $(OBJ_DIR)imagen.o $(OBJ_DIR)funciones.o $(OBJ_DIR)main_mnist_fc.o -o $(BIN_DIR)main
	#gcc -g -I$(HEADERS_DIR) -I$(IMG_DIR) -I$(PARAM_DIR) -Wall -O3 $(OBJ_DIR)imagen.o $(OBJ_DIR)funciones.o $(OBJ_DIR)main_mnist_cnn1.o -o $(BIN_DIR)main
	#gcc -g -I$(HEADERS_DIR) -I$(IMG_DIR) -I$(PARAM_DIR) -Wall -O3 $(OBJ_DIR)imagen.o $(OBJ_DIR)funciones.o $(OBJ_DIR)main_mnist_cnn2.o -o $(BIN_DIR)main
	gcc -g -I$(HEADERS_DIR) -I$(IMG_DIR) -I$(PARAM_DIR) -Wall -O3 $(OBJ_DIR)imagen.o $(OBJ_DIR)funciones.o $(OBJ_DIR)main_mnist_cnn13.o -o $(BIN_DIR)main

clean:
	-rm -f $(BIN_DIR)*
	-rm -f $(OBJ_DIR)*
