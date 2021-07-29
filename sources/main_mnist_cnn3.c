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
  // INICIALIZACIÓN SEMILLA ALEATORIA
  srand(time(NULL));

  unsigned int NUM_FILAS = 28;
  unsigned int NUM_COLUMNAS = 28;
  unsigned int NUM_CANALES = 1;
  unsigned int STEP_FILA = NUM_COLUMNAS*NUM_CANALES;
  unsigned int STEP_COLUMNA = NUM_CANALES;

  double time_taken;
  clock_t t;

  float *img_ptr = img;

  float *wc1_ptr = wc1;
  float *bc1_ptr = bc1;

  float *wn1_ptr = wn1;
  float *bn1_ptr = bn1;


  float *img_padding_2_por_cada_lado_ptr = (float *) malloc((NUM_FILAS+4) * (NUM_COLUMNAS+4) * NUM_CANALES * sizeof(float));
  float *img_conv_7x7_S2_ptr = (float *) malloc(16 * 16 * 3 * sizeof(float)); // 14x14x3 con padding de 1 => 16x16x3
  float *img_avg_pool_2x2_S2_ptr = (float *) malloc(9 * 9 * 3 * sizeof(float)); // 7x7x3 con padding de 1 => 9x9x3
  float *producto_tensores_ptr = (float *) malloc(10 * sizeof(float)); // Solo tenemos 10 neuronas


  // PADDING SAME
  unsigned int PADDING = 2;
  t = clock();
  img_padding_2_por_cada_lado_ptr = add_padding_same_con_copia_imagen(img_ptr, img_padding_2_por_cada_lado_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, PADDING);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  NUM_FILAS += PADDING*2;
  NUM_COLUMNAS += PADDING*2;
  NUM_CANALES += 0;
  STEP_FILA = NUM_COLUMNAS*NUM_CANALES;
  STEP_COLUMNA = NUM_CANALES;
  //visualizarImagen(img_padding_2_por_cada_lado_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  // CONV 7X7 S2
  unsigned int NUM_FILTROS = 3;
  unsigned int STEP_FILA_OUTPUT = 16*NUM_FILTROS; // 16 píxeles en una fila * canales==num_filtros

  t = clock();
  img_conv_7x7_S2_ptr = Convolucion_7x7_S2(img_padding_2_por_cada_lado_ptr, img_conv_7x7_S2_ptr, wc1_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, STEP_FILA_OUTPUT, NUM_FILTROS);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  PADDING = 1;
  NUM_FILAS = 16;
  NUM_COLUMNAS = 16;
  NUM_CANALES = NUM_FILTROS;
  STEP_FILA = NUM_COLUMNAS*NUM_CANALES;
  STEP_COLUMNA = NUM_CANALES;

  printf("\n======== conv_7x7_S2(IMAGEN INICIAL CON PADDING 2 POR CADA LADO, Wc1) ========\n");
  printf("Convolucion_7x7_S2() tardó %f segundos en ejecutarse \n", time_taken);
  //visualizarImagen(img_conv_7x7_S2_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  // SUMA DEL BIAS
  t = clock();
  img_conv_7x7_S2_ptr = suma_vector_con_tensor_por_canales(img_conv_7x7_S2_ptr, bc1_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, 1);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  printf("\n======== suma_vector_con_tensor_por_canales(conv_7x7_S2(IMAGEN INICIAL CON PADDING 2 POR CADA LADO, Wn1), bc1) ========\n");
  printf("suma_vector_con_tensor_por_canales() tardó %f segundos en ejecutarse \n", time_taken);
  //visualizarImagen(img_conv_7x7_S2_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  // ReLU
  t = clock();
  img_conv_7x7_S2_ptr = ReLU(img_conv_7x7_S2_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, PADDING);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  printf("\n======== ReLU(suma_vector_con_tensor_por_canales(conv_7x7_S2(IMAGEN INICIAL CON PADDING 2 POR CADA LADO, Wn1), bc1)) ========\n");
  printf("ReLU() tardó %f segundos en ejecutarse \n", time_taken);
  //visualizarImagen(img_conv_7x7_S2_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  // Avg Pool
  t = clock();
  img_avg_pool_2x2_S2_ptr = Avg_Pooling_2x2_S2(img_conv_7x7_S2_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, PADDING, 1, img_avg_pool_2x2_S2_ptr);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  NUM_FILAS = (NUM_FILAS-PADDING*2)/2;
  NUM_COLUMNAS = (NUM_COLUMNAS-PADDING*2)/2;
  PADDING = 1;
  NUM_FILAS += PADDING*2;
  NUM_COLUMNAS += PADDING*2;
  NUM_CANALES += 0;
  STEP_FILA = NUM_COLUMNAS*NUM_CANALES;
  STEP_COLUMNA = NUM_CANALES;

  printf("\n======== Avg_Pooling_2x2_S2(ReLU(suma_vector_con_tensor_por_canales(conv_7x7_S2(IMAGEN INICIAL CON PADDING 2 POR CADA LADO, Wn1), bc1))) ========\n");
  printf("Avg_Pooling_2x2_S2() tardó %f segundos en ejecutarse \n", time_taken);
  //visualizarImagen(img_avg_pool_2x2_S2_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  // PRODUCTO TENSORES
  t = clock();
  producto_tensores_ptr = producto_tensores(img_avg_pool_2x2_S2_ptr, wn1_ptr, producto_tensores_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, 10);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  printf("\n======== producto_tensores(Avg_Pooling_2x2_S2(ReLU(suma_vector_con_tensor_por_canales(conv_7x7_S2(IMAGEN INICIAL CON PADDING 2 POR CADA LADO, Wn1), bc1)))) ========\n");
  printf("producto_tensores() tardó %f segundos en ejecutarse \n", time_taken);
  //visualizarImagen(producto_tensores_ptr, 1, 1, 10, 1, 1);


  // SUMA DEL BIAS
  t = clock();
  producto_tensores_ptr = suma_vector_con_tensor_por_canales(producto_tensores_ptr, bn1_ptr, 1, 1, 10, 1, 1, 0);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  printf("\n======== suma_vector_con_tensor_por_canales(producto_tensores(Avg_Pooling_2x2_S2(ReLU(suma_vector_con_tensor_por_canales(conv_7x7_S2(IMAGEN INICIAL CON PADDING 2 POR CADA LADO, Wn1), bc1))))) ========\n");
  printf("suma_vector_con_tensor_por_canales() tardó %f segundos en ejecutarse \n", time_taken);
  visualizarImagen(producto_tensores_ptr, 1, 1, 10, 1, 1);

  getchar();

  return 0;
}
