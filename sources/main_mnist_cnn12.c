#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

#include "imagen.h"
#include "funciones.h"

#include "imagen_0_de_test.h"

#include "wc1.h"
#include "bc1.h"

#include "wc2.h"
#include "bc2.h"

#include "wc3.h"
#include "bc3.h"

#include "wc4.h"
#include "bc4.h"

#include "wc5.h"
#include "bc5.h"

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
  unsigned int PADDING = -1;
  unsigned int NUM_FILTROS = -1;
  unsigned int STEP_FILA_OUTPUT = -1;
  unsigned int STEP_COLUMNA_OUTPUT = -1;

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

  float *wc2_ptr = wc2;
  float *bc2_ptr = bc2;

  float *wc3_ptr = wc3;
  float *bc3_ptr = bc3;

  float *wc4_ptr = wc4;
  float *bc4_ptr = bc4;

  float *wc5_ptr = wc5;
  float *bc5_ptr = bc5;

  float *wn1_ptr = wn1;
  float *bn1_ptr = bn1;

  //---------------------------------------------------------------------------------------------------
  // Estructuras de datos
  //---------------------------------------------------------------------------------------------------
  float *img_padding_3_por_cada_lado_ptr = (float *) calloc(34 * 34 * 1, sizeof(float));

  float *img_conv1_ptr = (float *) calloc(16 * 16 * 8, sizeof(float));
  float *img_conv2_ptr = (float *) calloc(16 * 16 * 12, sizeof(float));
  float *img_conv3_ptr = (float *) calloc(16 * 16 * 8, sizeof(float));
  float *img_conv4_ptr = (float *) calloc(9 * 9 * 16, sizeof(float));
  float *img_conv5_ptr = (float *) calloc(9 * 9 * 16, sizeof(float));

  float *img_avg_global_ptr = (float *) calloc(16, sizeof(float));

  float *producto_tensores_ptr = (float *) calloc(10, sizeof(float)); // Solo tenemos 10 neuronas

  printf("Tam entrada: [%d,%d,%d]\n", NUM_FILAS, NUM_COLUMNAS, NUM_CANALES);
  //---------------------------------------------------------------------------------------------------
  // PADDING SAME
  //---------------------------------------------------------------------------------------------------
  PADDING = 3;
  t = clock();
  img_padding_3_por_cada_lado_ptr = add_padding_same_con_copia_imagen(img_ptr, img_padding_3_por_cada_lado_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, PADDING);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  NUM_FILAS += PADDING*2;
  NUM_COLUMNAS += PADDING*2;
  NUM_CANALES += 0;
  STEP_FILA = NUM_COLUMNAS*NUM_CANALES;
  STEP_COLUMNA = NUM_CANALES;

  printf("\n======== Padding (img, 3) ========\n");
  printf("Tam salida: [%d,%d,%d]\n", NUM_FILAS, NUM_COLUMNAS, NUM_CANALES);
  printf("img_padding_1_por_cada_lado_ptr() tardó %f segundos en ejecutarse \n", time_taken);
  //visualizarImagen(img_padding_3_por_cada_lado_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  //---------------------------------------------------------------------------------------------------
  // CONV_1 (7X7 S2)
  //---------------------------------------------------------------------------------------------------
  NUM_FILTROS = 8;
  STEP_FILA_OUTPUT = 16*NUM_FILTROS;

  t = clock();
  img_conv1_ptr = Convolucion_7x7_S2(img_padding_3_por_cada_lado_ptr, img_conv1_ptr, wc1_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, STEP_FILA_OUTPUT, NUM_FILTROS);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  PADDING = 1;
  NUM_FILAS = 16;
  NUM_COLUMNAS = 16;
  NUM_CANALES = NUM_FILTROS;
  STEP_FILA = NUM_COLUMNAS*NUM_CANALES;
  STEP_COLUMNA = NUM_CANALES;

  printf("\n======== CONV_1 (7x7 S2) ========\n");
  printf("Tam salida: [%d,%d,%d]\n", NUM_FILAS, NUM_COLUMNAS, NUM_CANALES);
  printf("Convolucion_7x7_S2() tardó %f segundos en ejecutarse \n", time_taken);
  //visualizarImagen(img_conv1_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  // SUMA DEL BIAS
  t = clock();
  img_conv1_ptr = suma_vector_con_tensor_por_canales(img_conv1_ptr, bc1_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, LA_ENTRADA_TIENE_PADDING);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  printf("\n======== CONV_1 (7x7 S2) + B ========\n");
  printf("Tam salida: [%d,%d,%d]\n", NUM_FILAS, NUM_COLUMNAS, NUM_CANALES);
  printf("suma_vector_con_tensor_por_canales() tardó %f segundos en ejecutarse \n", time_taken);
  //visualizarImagen(img_conv1_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  // ReLU
  t = clock();
  img_conv1_ptr = ReLU(img_conv1_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, PADDING);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  printf("\n======== ReLU(CONV_1 (7x7 S2) + B) ========\n");
  printf("Tam salida: [%d,%d,%d]\n", NUM_FILAS, NUM_COLUMNAS, NUM_CANALES);
  printf("ReLU() tardó %f segundos en ejecutarse \n", time_taken);
  //visualizarImagen(img_conv1_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  //---------------------------------------------------------------------------------------------------
  // CONV_2 (3X3 S1)
  //---------------------------------------------------------------------------------------------------
  NUM_FILTROS = 12;
  STEP_FILA_OUTPUT = NUM_COLUMNAS*NUM_FILTROS;
  STEP_COLUMNA_OUTPUT = NUM_FILTROS;

  t = clock();
  img_conv2_ptr = Convolucion_3x3_S1(img_conv1_ptr, img_conv2_ptr, wc2_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, STEP_FILA_OUTPUT, STEP_COLUMNA_OUTPUT, NUM_FILTROS);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  NUM_CANALES = NUM_FILTROS;
  STEP_FILA = NUM_COLUMNAS*NUM_CANALES;
  STEP_COLUMNA = NUM_CANALES;

  printf("\n======== CONV_2 (3x3 S1) ========\n");
  printf("Tam salida: [%d,%d,%d]\n", NUM_FILAS, NUM_COLUMNAS, NUM_CANALES);
  printf("Convolucion_3x3_S1() tardó %f segundos en ejecutarse \n", time_taken);
  //visualizarImagen(img_conv2_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  // SUMA DEL BIAS
  t = clock();
  img_conv2_ptr = suma_vector_con_tensor_por_canales(img_conv2_ptr, bc2_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, LA_ENTRADA_TIENE_PADDING);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  printf("\n======== CONV_2 (3x3 S1) + B ========\n");
  printf("Tam salida: [%d,%d,%d]\n", NUM_FILAS, NUM_COLUMNAS, NUM_CANALES);
  printf("suma_vector_con_tensor_por_canales() tardó %f segundos en ejecutarse \n", time_taken);
  //visualizarImagen(img_conv2_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  // ReLU
  t = clock();
  img_conv2_ptr = ReLU(img_conv2_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, PADDING);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  printf("\n======== ReLU(CONV_2 (3x3 S1) + B) ========\n");
  printf("Tam salida: [%d,%d,%d]\n", NUM_FILAS, NUM_COLUMNAS, NUM_CANALES);
  printf("ReLU() tardó %f segundos en ejecutarse \n", time_taken);
  //visualizarImagen(img_conv2_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  //---------------------------------------------------------------------------------------------------
  // CONV_3 (1X1 S1)
  //---------------------------------------------------------------------------------------------------
  NUM_FILAS = 16;
  NUM_COLUMNAS = 16;
  NUM_CANALES = 8;
  STEP_FILA = NUM_COLUMNAS*NUM_CANALES;
  STEP_COLUMNA = NUM_CANALES;

  NUM_FILTROS = 8;
  STEP_FILA_OUTPUT = NUM_COLUMNAS*NUM_FILTROS;
  STEP_COLUMNA_OUTPUT = NUM_FILTROS;

  t = clock();
  img_conv3_ptr = Convolucion_1x1_S1(img_conv1_ptr, img_conv3_ptr, wc3_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, STEP_FILA_OUTPUT, STEP_COLUMNA_OUTPUT, NUM_FILTROS);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  NUM_CANALES = NUM_FILTROS;
  STEP_FILA = NUM_COLUMNAS*NUM_CANALES;
  STEP_COLUMNA = NUM_CANALES;

  printf("\n======== CONV_3 (1x1 S1) ========\n");
  printf("Tam salida: [%d,%d,%d]\n", NUM_FILAS, NUM_COLUMNAS, NUM_CANALES);
  printf("Convolucion_1x1_S1() tardó %f segundos en ejecutarse \n", time_taken);
  //visualizarImagen(img_conv3_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  // SUMA DEL BIAS
  t = clock();
  img_conv3_ptr = suma_vector_con_tensor_por_canales(img_conv3_ptr, bc3_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, LA_ENTRADA_TIENE_PADDING);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  printf("\n======== CONV_3 (1x1 S1) + B ========\n");
  printf("Tam salida: [%d,%d,%d]\n", NUM_FILAS, NUM_COLUMNAS, NUM_CANALES);
  printf("suma_vector_con_tensor_por_canales() tardó %f segundos en ejecutarse \n", time_taken);
  //visualizarImagen(img_conv3_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  // ReLU
  t = clock();
  img_conv3_ptr = ReLU(img_conv3_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, PADDING);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  printf("\n======== ReLU(CONV_3 (1x1 S1) + B) ========\n");
  printf("Tam salida: [%d,%d,%d]\n", NUM_FILAS, NUM_COLUMNAS, NUM_CANALES);
  printf("ReLU() tardó %f segundos en ejecutarse \n", time_taken);
  //visualizarImagen(img_conv3_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  //---------------------------------------------------------------------------------------------------
  // CONV_4 (1X1 S2)
  //---------------------------------------------------------------------------------------------------
  NUM_FILAS = 16;
  NUM_COLUMNAS = 16;
  NUM_CANALES = 12;
  STEP_FILA = NUM_COLUMNAS*NUM_CANALES;
  STEP_COLUMNA = NUM_CANALES;

  NUM_FILTROS = 16;
  STEP_FILA_OUTPUT = 9*NUM_FILTROS;
  STEP_COLUMNA_OUTPUT = NUM_FILTROS;

  t = clock();
  img_conv4_ptr = Convolucion_1x1_S2(img_conv2_ptr, img_conv4_ptr, wc4_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, STEP_FILA_OUTPUT, STEP_COLUMNA_OUTPUT, NUM_FILTROS);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  NUM_FILAS = 9;
  NUM_COLUMNAS = 9;
  NUM_CANALES = NUM_FILTROS;
  STEP_FILA = NUM_COLUMNAS*NUM_CANALES;
  STEP_COLUMNA = NUM_CANALES;

  printf("\n======== CONV_4 (1x1 S2) ========\n");
  printf("Tam salida: [%d,%d,%d]\n", NUM_FILAS, NUM_COLUMNAS, NUM_CANALES);
  printf("Convolucion_1x1_S2() tardó %f segundos en ejecutarse \n", time_taken);
  //visualizarImagen(img_conv4_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  // SUMA DEL BIAS
  t = clock();
  img_conv4_ptr = suma_vector_con_tensor_por_canales(img_conv4_ptr, bc4_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, LA_ENTRADA_TIENE_PADDING);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  printf("\n======== CONV_4 (1x1 S2) + B ========\n");
  printf("Tam salida: [%d,%d,%d]\n", NUM_FILAS, NUM_COLUMNAS, NUM_CANALES);
  printf("suma_vector_con_tensor_por_canales() tardó %f segundos en ejecutarse \n", time_taken);
  //visualizarImagen(img_conv4_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  // ReLU
  t = clock();
  img_conv4_ptr = ReLU(img_conv4_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, PADDING);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  printf("\n======== ReLU(CONV_4 (1x1 S2) + B) ========\n");
  printf("Tam salida: [%d,%d,%d]\n", NUM_FILAS, NUM_COLUMNAS, NUM_CANALES);
  printf("ReLU() tardó %f segundos en ejecutarse \n", time_taken);
  //visualizarImagen(img_conv4_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  //---------------------------------------------------------------------------------------------------
  // CONV_5 (3X3 S2)
  //---------------------------------------------------------------------------------------------------
  NUM_FILAS = 16;
  NUM_COLUMNAS = 16;
  NUM_CANALES = 8;
  STEP_FILA = NUM_COLUMNAS*NUM_CANALES;
  STEP_COLUMNA = NUM_CANALES;
  NUM_FILTROS = 16;
  STEP_FILA_OUTPUT = 9*NUM_FILTROS;
  STEP_COLUMNA_OUTPUT = NUM_FILTROS;

  t = clock();
  img_conv5_ptr = Convolucion_3x3_S2(img_conv3_ptr, img_conv5_ptr, wc5_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, STEP_FILA_OUTPUT, STEP_COLUMNA_OUTPUT, NUM_FILTROS);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  NUM_FILAS = 9;
  NUM_COLUMNAS = 9;
  NUM_CANALES = NUM_FILTROS;
  STEP_FILA = NUM_COLUMNAS*NUM_CANALES;
  STEP_COLUMNA = NUM_CANALES;

  printf("\n======== CONV_5 (3x3 S2) ========\n");
  printf("Tam salida: [%d,%d,%d]\n", NUM_FILAS, NUM_COLUMNAS, NUM_CANALES);
  printf("Convolucion_3x3_S2() tardó %f segundos en ejecutarse \n", time_taken);
  //visualizarImagen(img_conv4_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  // SUMA DEL BIAS
  t = clock();
  img_conv5_ptr = suma_vector_con_tensor_por_canales(img_conv5_ptr, bc5_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, LA_ENTRADA_TIENE_PADDING);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  printf("\n======== CONV_5 (3x3 S2) + B ========\n");
  printf("Tam salida: [%d,%d,%d]\n", NUM_FILAS, NUM_COLUMNAS, NUM_CANALES);
  printf("suma_vector_con_tensor_por_canales() tardó %f segundos en ejecutarse \n", time_taken);
  //visualizarImagen(img_conv5_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  // ReLU
  t = clock();
  img_conv5_ptr = ReLU(img_conv5_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, PADDING);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  printf("\n======== ReLU(CONV_5 (3x3 S2) + B) ========\n");
  printf("Tam salida: [%d,%d,%d]\n", NUM_FILAS, NUM_COLUMNAS, NUM_CANALES);
  printf("ReLU() tardó %f segundos en ejecutarse \n", time_taken);
  //visualizarImagen(img_conv5_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  //---------------------------------------------------------------------------------------------------
  // SUMA
  //---------------------------------------------------------------------------------------------------
  t = clock();
  img_conv4_ptr = suma_elemento_elemento(img_conv4_ptr, img_conv5_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  printf("\n======== CONV_4 + CONV_5 ========\n");
  printf("Tam salida: [%d,%d,%d]\n", NUM_FILAS, NUM_COLUMNAS, NUM_CANALES);
  printf("suma_elemento_elemento() tardó %f segundos en ejecutarse \n", time_taken);
  //visualizarImagen(img_conv4_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  //---------------------------------------------------------------------------------------------------
  // AVG GLOBAL POOLING
  //---------------------------------------------------------------------------------------------------
  t = clock();
  img_avg_global_ptr = Avg_Pooling_Completo_por_canal(img_conv4_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, PADDING, img_avg_global_ptr);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  NUM_FILAS = 3;
  NUM_COLUMNAS = 3;
  STEP_FILA = NUM_COLUMNAS*NUM_CANALES;
  STEP_COLUMNA = NUM_CANALES;

  printf("\n======== Avg_Pooling_Completo_por_canal(CONV_4 + CONV_5) ========\n");
  printf("Tam salida: [%d,%d,%d]\n", NUM_FILAS, NUM_COLUMNAS, NUM_CANALES);
  printf("Avg_Pooling_Completo_por_canal() tardó %f segundos en ejecutarse \n", time_taken);
  //visualizarImagen(img_avg_global_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  //---------------------------------------------------------------------------------------------------
  // PRODUCTO TENSORES
  //---------------------------------------------------------------------------------------------------
  t = clock();
  producto_tensores_ptr = producto_tensores(img_avg_global_ptr, wn1_ptr, producto_tensores_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, NUM_NEURONAS);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  NUM_FILAS = 1;
  NUM_COLUMNAS = 1;
  NUM_CANALES = NUM_NEURONAS;
  STEP_FILA = NUM_COLUMNAS*NUM_CANALES;
  STEP_COLUMNA = NUM_CANALES;

  printf("\n======== producto_tensores(AVG_GLOBAL_POOL) ========\n");
  printf("Tam salida: [%d,%d,%d]\n", NUM_FILAS, NUM_COLUMNAS, NUM_CANALES);
  printf("producto_tensores() tardó %f segundos en ejecutarse \n", time_taken);
  //visualizarImagen(producto_tensores_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  // SUMA DEL BIAS
  t = clock();
  producto_tensores_ptr = suma_vector_con_tensor_por_canales(producto_tensores_ptr, bn1_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, LA_ENTRADA_NO_TIENE_PADDING);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  printf("\n======== producto_tensores(AVG_GLOBAL_POOL) + B ========\n");
  printf("Tam salida: [%d,%d,%d]\n", NUM_FILAS, NUM_COLUMNAS, NUM_CANALES);
  printf("suma_vector_con_tensor_por_canales() tardó %f segundos en ejecutarse \n", time_taken);
  visualizarImagen(producto_tensores_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);

  //getchar();

  return 0;
}
