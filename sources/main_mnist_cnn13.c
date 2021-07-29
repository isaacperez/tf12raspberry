#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

#include "imagen.h"
#include "funciones.h"

#include "imagen_0_de_test.h"

#include "wc1.h"
#include "bc1.h"

#include "wn1.h"
#include "bn1.h"


/*****************************************************************
* MAIN()
******************************************************************/
int main()
{
  //---------------------------------------------------------------------------------------------------
  // INICIALIZACIÓN SEMILLA ALEATORIA
  //---------------------------------------------------------------------------------------------------
  srand(time(NULL));

  //---------------------------------------------------------------------------------------------------
  // Variables globales del programa
  //---------------------------------------------------------------------------------------------------
  unsigned int NUM_FILAS = 28;
  unsigned int NUM_COLUMNAS = 28;
  unsigned int NUM_CANALES = 1;
  unsigned int STEP_FILA = NUM_COLUMNAS*NUM_CANALES;
  unsigned int STEP_COLUMNA = NUM_CANALES;
  unsigned int NUM_NEURONAS = 10;

  const unsigned int LA_ENTRADA_NO_TIENE_PADDING = 0;
  const unsigned int LA_ENTRADA_TIENE_PADDING = 1;

  double time_taken;
  clock_t t;

  //---------------------------------------------------------------------------------------------------
  // Datos
  //---------------------------------------------------------------------------------------------------
  float *img_ptr = img;

  float *wc1_ptr = wc1;
  float *bc1_ptr = bc1;

  float *wn1_ptr = wn1;
  float *bn1_ptr = bn1;

  //---------------------------------------------------------------------------------------------------
  // Estructuras de datos
  //---------------------------------------------------------------------------------------------------
  float *img_padding_1_por_cada_lado_ptr = (float *) calloc(30 * 30 * 1, sizeof(float));
  float *img_conv1_ptr = (float *) calloc(30 * 30 * 4, sizeof(float));

  float *img_max_pool_ptr = (float *) calloc(16 * 16 * 4, sizeof(float));

  float *producto_tensores_ptr = (float *) calloc(10, sizeof(float)); // Solo tenemos 10 neuronas

  printf("Tam entrada: [%d,%d,%d]\n", NUM_FILAS, NUM_COLUMNAS, NUM_CANALES);

  //---------------------------------------------------------------------------------------------------
  // PADDING SAME
  //---------------------------------------------------------------------------------------------------
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

  printf("\n======== Padding (img, 1) ========\n");
  printf("Tam salida: [%d,%d,%d]\n", NUM_FILAS, NUM_COLUMNAS, NUM_CANALES);
  printf("add_padding_same_con_copia_imagen() tardó %f segundos en ejecutarse \n", time_taken);
  //visualizarImagen(img_padding_1_por_cada_lado_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);

  //---------------------------------------------------------------------------------------------------
  // CONV_1 (1X1 S1)
  //---------------------------------------------------------------------------------------------------
  unsigned int NUM_FILTROS = 4;
  unsigned int STEP_FILA_OUTPUT = NUM_COLUMNAS*NUM_FILTROS;
  unsigned int STEP_COLUMNA_OUTPUT = NUM_FILTROS;

  t = clock();
  img_conv1_ptr = Convolucion_1x1_S1(img_padding_1_por_cada_lado_ptr, img_conv1_ptr, wc1_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, STEP_FILA_OUTPUT, STEP_COLUMNA_OUTPUT, NUM_FILTROS);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  NUM_CANALES = NUM_FILTROS;
  STEP_FILA = NUM_COLUMNAS*NUM_CANALES;
  STEP_COLUMNA = NUM_CANALES;

  printf("\n======== CONV_1 (1X1 S1) ========\n");
  printf("Tam salida: [%d,%d,%d]\n", NUM_FILAS, NUM_COLUMNAS, NUM_CANALES);
  printf("Convolucion_1x1_S1() tardó %f segundos en ejecutarse \n", time_taken);
  //visualizarImagen(img_conv1_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);

  /*
  float maximo = *img_conv1_ptr;
  for (unsigned int i= 0; i < (30*30*4); i++){
    maximo = (maximo > *(img_conv1_ptr + i)) ? maximo : *(img_conv1_ptr + i);
  }
  printf("Max: %5.8f", maximo);
  */

  // SUMA DEL BIAS
  t = clock();
  img_conv1_ptr = suma_vector_con_tensor_por_canales(img_conv1_ptr, bc1_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, LA_ENTRADA_TIENE_PADDING);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  printf("\n======== CONV_1 (1X1 S1) + B ========\n");
  printf("Tam salida: [%d,%d,%d]\n", NUM_FILAS, NUM_COLUMNAS, NUM_CANALES);
  printf("suma_vector_con_tensor_por_canales() tardó %f segundos en ejecutarse \n", time_taken);
  //visualizarImagen(img_conv1_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  // ReLU
  t = clock();
  img_conv1_ptr = ReLU(img_conv1_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, PADDING);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  printf("\n======== ReLU(CONV_1 (1X1 S1) + B) ========\n");
  printf("Tam salida: [%d,%d,%d]\n", NUM_FILAS, NUM_COLUMNAS, NUM_CANALES);
  printf("ReLU() tardó %f segundos en ejecutarse \n", time_taken);
  visualizarImagen(img_conv1_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  //---------------------------------------------------------------------------------------------------
  // MAX POOLING
  //---------------------------------------------------------------------------------------------------
  STEP_FILA_OUTPUT = 16*NUM_CANALES;
  t = clock();
  img_max_pool_ptr = Max_Pooling_2x2_S2(img_conv1_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, STEP_FILA_OUTPUT, PADDING, PADDING, img_max_pool_ptr);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  NUM_FILAS = 16;
  NUM_COLUMNAS = 16;
  NUM_CANALES += 0;
  STEP_FILA = NUM_COLUMNAS*NUM_CANALES;
  STEP_COLUMNA = NUM_CANALES;

  printf("\n======== Max_Pooling_3x3(Conv_1) ========\n");
  printf("Tam salida: [%d,%d,%d]\n", NUM_FILAS, NUM_COLUMNAS, NUM_CANALES);
  printf("Max_Pooling_3x2_S3() tardó %f segundos en ejecutarse \n", time_taken);
  visualizarImagen(img_max_pool_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);

  //---------------------------------------------------------------------------------------------------
  // PRODUCTO TENSORES
  //---------------------------------------------------------------------------------------------------
  t = clock();
  producto_tensores_ptr = producto_tensores(img_max_pool_ptr, wn1_ptr, producto_tensores_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, NUM_NEURONAS);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  NUM_FILAS = 1;
  NUM_COLUMNAS = 1;
  NUM_CANALES = NUM_NEURONAS;
  STEP_FILA = NUM_COLUMNAS*NUM_CANALES;
  STEP_COLUMNA = NUM_CANALES;

  printf("\n======== producto_tensores(Max_Pooling_3x3(Conv_1)) ========\n");
  printf("Tam salida: [%d,%d,%d]\n", NUM_FILAS, NUM_COLUMNAS, NUM_CANALES);
  printf("producto_tensores() tardó %f segundos en ejecutarse \n", time_taken);
  visualizarImagen(producto_tensores_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  // SUMA DEL BIAS
  t = clock();
  producto_tensores_ptr = suma_vector_con_tensor_por_canales(producto_tensores_ptr, bn1_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, LA_ENTRADA_NO_TIENE_PADDING);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  printf("\n======== producto_tensores(Max_Pooling_3x3(Conv_1)) + B ========\n");
  printf("Tam salida: [%d,%d,%d]\n", NUM_FILAS, NUM_COLUMNAS, NUM_CANALES);
  printf("suma_vector_con_tensor_por_canales() tardó %f segundos en ejecutarse \n", time_taken);
  visualizarImagen(producto_tensores_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);

  //getchar();

  return 0;
}
