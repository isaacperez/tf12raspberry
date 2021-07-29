#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

#include "imagen.h"
#include "funciones.h"

#include "imagen_0_de_test.h"
#include "b.h"
#include "w.h"

/*****************************************************************
* MAIN()
******************************************************************/
int main()
{
  // INICIALIZACIÓN SEMILLA ALEATORIA
  srand(time(NULL));

  unsigned int NUM_FILAS = 28;
  unsigned int NUM_COLUMNAS = 28;
  unsigned int NUM_CANALES = 1;
  unsigned int STEP_FILA = NUM_COLUMNAS*NUM_CANALES;
  unsigned int STEP_COLUMNA = NUM_CANALES;

  double time_taken;
  clock_t t;

  printf("Hello World!\n");
  float *img_ptr = img;
  float *W_ptr = W;
  float *B_ptr = B;
  float *img_padding_1_por_cada_lado_ptr = (float *) malloc((NUM_FILAS+2) * (NUM_COLUMNAS+2) * NUM_CANALES * sizeof(float));
  float *producto_tensores_ptr = (float *) malloc(10 * sizeof(float)); // Solo tenemos 10 neuronas


  // PADDING SAME
  unsigned int PADDING = 1;
  t = clock();
  img_padding_1_por_cada_lado_ptr = add_padding_same_con_copia_imagen(img_ptr, img_padding_1_por_cada_lado_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, PADDING);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  NUM_FILAS += PADDING*2;
  NUM_COLUMNAS += PADDING*2;
  NUM_CANALES += 0;
  STEP_FILA = NUM_COLUMNAS*NUM_CANALES;
  STEP_COLUMNA = NUM_CANALES;

  //visualizarImagen(img_padding_1_por_cada_lado_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  // PRODUCTO TENSORES
  t = clock();
  producto_tensores_ptr = producto_tensores(img_padding_1_por_cada_lado_ptr, W_ptr, producto_tensores_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, 10);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  printf("\n======== producto_tensores(IMAGEN INICIAL CON PADDING 1 POR CADA LADO, W) ========\n");
  printf("producto_tensores() tardó %f segundos en ejecutarse \n", time_taken);
  visualizarImagen(producto_tensores_ptr, 1, 1, 10, 1, 1);

  // SUMA DEL BIAS
  t = clock();
  producto_tensores_ptr = suma_vector_con_tensor_por_canales(producto_tensores_ptr, B_ptr, 1, 1, 10, 1, 1, 0);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  printf("\n======== suma_vector_con_tensor_por_canales(producto_tensores(IMAGEN INICIAL CON PADDING 1 POR CADA LADO)), B) ========\n");
  printf("suma_vector_con_tensor_por_canales() tardó %f segundos en ejecutarse \n", time_taken);
  visualizarImagen(producto_tensores_ptr, 1, 1, 10, 1, 1);

  getchar();

  return 0;
}
