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

  float *wc2_ptr = wc2;
  float *bc2_ptr = bc2;

  float *wn1_ptr = wn1;
  float *bn1_ptr = bn1;


  float *img_padding_1_por_cada_lado_ptr = (float *) malloc((NUM_FILAS+2) * (NUM_COLUMNAS+2) * NUM_CANALES * sizeof(float));
  float *img_conv_1x1_S1_ptr = (float *) malloc((NUM_FILAS+2) * (NUM_COLUMNAS+2) * 3 * sizeof(float));
  float *img_segunda_conv_ptr = (float *) malloc((NUM_FILAS+2) * (NUM_COLUMNAS+2)*2* sizeof(float));
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


  // CONV 1X1 S1
  unsigned int NUM_FILTROS = 3;
  unsigned int STEP_FILA_OUTPUT = NUM_COLUMNAS*NUM_FILTROS; // 32 píxeles en una fila * canales==num_filtros
  unsigned int STEP_COLUMNA_OUTPUT = NUM_FILTROS; // 32 píxeles en una fila * canales==num_filtros

  t = clock();
  img_conv_1x1_S1_ptr = Convolucion_1x1_S1(img_padding_1_por_cada_lado_ptr, img_conv_1x1_S1_ptr, wc1_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, STEP_FILA_OUTPUT, STEP_COLUMNA_OUTPUT, NUM_FILTROS);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  NUM_CANALES = NUM_FILTROS;
  STEP_FILA = NUM_COLUMNAS*NUM_CANALES;
  STEP_COLUMNA = NUM_CANALES;

  printf("\n======== Convolucion_1x1_S1(IMAGEN INICIAL CON PADDING 1 POR CADA LADO, Wc1) ========\n");
  printf("Convolucion_1x1_S1() tardó %f segundos en ejecutarse \n", time_taken);
  //visualizarImagen(img_conv_1x1_S1_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  // SUMA DEL BIAS
  t = clock();
  img_conv_1x1_S1_ptr = suma_vector_con_tensor_por_canales(img_conv_1x1_S1_ptr, bc1_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, 1);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  printf("\n======== suma_vector_con_tensor_por_canales(Convolucion_1x1_S1(IMAGEN INICIAL CON PADDING 1 POR CADA LADO, Wc1), bc1) ========\n");
  printf("suma_vector_con_tensor_por_canales() tardó %f segundos en ejecutarse \n", time_taken);
  //visualizarImagen(img_conv_1x1_S1_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  // ReLU
  t = clock();
  img_conv_1x1_S1_ptr = ReLU(img_conv_1x1_S1_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, PADDING);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  printf("\n======== ReLU(suma_vector_con_tensor_por_canales(Convolucion_1x1_S1(IMAGEN INICIAL CON PADDING 1 POR CADA LADO, Wc1), bc1)) ========\n");
  printf("ReLU() tardó %f segundos en ejecutarse \n", time_taken);
  //visualizarImagen(img_conv_1x1_S1_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  // CONV 3X3 S1
  NUM_FILTROS = 2;
  STEP_FILA_OUTPUT = NUM_COLUMNAS*NUM_FILTROS; // 30 píxeles en una fila * canales==num_filtros
  STEP_COLUMNA_OUTPUT = NUM_FILTROS;

  t = clock();
  img_segunda_conv_ptr = Convolucion_3x3_S1(img_conv_1x1_S1_ptr, img_segunda_conv_ptr, wc2_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, STEP_FILA_OUTPUT, STEP_COLUMNA_OUTPUT, NUM_FILTROS);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  NUM_CANALES = NUM_FILTROS;
  STEP_FILA = NUM_COLUMNAS*NUM_CANALES;
  STEP_COLUMNA = NUM_CANALES;

  printf("\n======== Convolucion_3x3_S1(ReLU(suma_vector_con_tensor_por_canales(Convolucion_1x1_S1(IMAGEN INICIAL CON PADDING 1 POR CADA LADO, Wc1), bc1)), Wc2) ========\n");
  printf("Convolucion_3x3_S1() tardó %f segundos en ejecutarse \n", time_taken);
  visualizarImagen(img_segunda_conv_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  // SUMA DEL BIAS
  t = clock();
  img_segunda_conv_ptr = suma_vector_con_tensor_por_canales(img_segunda_conv_ptr, bc2_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, 1);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  printf("\n======== suma_vector_con_tensor_por_canales(Convolucion_3x3_S1(ReLU(suma_vector_con_tensor_por_canales(Convolucion_1x1_S1(IMAGEN INICIAL CON PADDING 1 POR CADA LADO, Wc1), bc1)), Wc2), bc2) ========\n");
  printf("suma_vector_con_tensor_por_canales() tardó %f segundos en ejecutarse \n", time_taken);
  //visualizarImagen(img_segunda_conv_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  // ReLU
  t = clock();
  img_segunda_conv_ptr = ReLU(img_segunda_conv_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, PADDING);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  printf("\n======== ReLU(suma_vector_con_tensor_por_canales(Convolucion_3x3_S1(ReLU(suma_vector_con_tensor_por_canales(Convolucion_1x1_S1(IMAGEN INICIAL CON PADDING 1 POR CADA LADO, Wc1), bc1)), Wc2), bc2)) ========\n");
  printf("ReLU() tardó %f segundos en ejecutarse \n", time_taken);
  //visualizarImagen(img_segunda_conv_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  // PRODUCTO TENSORES
  t = clock();
  producto_tensores_ptr = producto_tensores(img_segunda_conv_ptr, wn1_ptr, producto_tensores_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, 10);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  printf("\n======== producto_tensores(ReLU(suma_vector_con_tensor_por_canales(Convolucion_3x3_S1(ReLU(suma_vector_con_tensor_por_canales(Convolucion_1x1_S1(IMAGEN INICIAL CON PADDING 1 POR CADA LADO, Wc1), bc1)), Wc2), bc2)), Wn1) ========\n");
  printf("producto_tensores() tardó %f segundos en ejecutarse \n", time_taken);
  //visualizarImagen(producto_tensores_ptr, 1, 1, 10, 1, 1);


  // SUMA DEL BIAS
  t = clock();
  producto_tensores_ptr = suma_vector_con_tensor_por_canales(producto_tensores_ptr, bn1_ptr, 1, 1, 10, 1, 1, 0);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  printf("\n======== suma_vector_con_tensor_por_canales(producto_tensores(ReLU(suma_vector_con_tensor_por_canales(Convolucion_3x3_S1(ReLU(suma_vector_con_tensor_por_canales(Convolucion_1x1_S1(IMAGEN INICIAL CON PADDING 1 POR CADA LADO, Wc1), bc1)), Wc2), bc2)), Wn1)) ========\n");
  printf("suma_vector_con_tensor_por_canales() tardó %f segundos en ejecutarse \n", time_taken);
  visualizarImagen(producto_tensores_ptr, 1, 1, 10, 1, 1);

  getchar();

  return 0;
}
