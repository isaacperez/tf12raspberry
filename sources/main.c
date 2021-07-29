#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

#include "imagen.h"
#include "funciones.h"

/*****************************************************************
* MAIN()
******************************************************************/
int main()
{
  // INICIALIZACIÓN SEMILLA ALEATORIA
  srand(time(NULL));


  // VARIABLES GLOBALES DEL PROGRAMA != CONV 7X7
  unsigned int NUM_FILAS = 6;
  unsigned int NUM_COLUMNAS = 6;
  unsigned int NUM_CANALES = 2;
  unsigned int STEP_FILA = NUM_COLUMNAS*NUM_CANALES;
  unsigned int STEP_COLUMNA = NUM_CANALES;

  double time_taken;
  clock_t t;

  if (NUM_FILAS % 2 != 0 || NUM_COLUMNAS % 2 != 0){
    printf("La imagen debe tener un número par de filas y columnas\n");
    exit(0);
  }


  // CREACION DE ESPACIOS DE MEMORIA
  type_for_pixels *img_ptr = (type_for_pixels *) malloc(NUM_FILAS * NUM_COLUMNAS * NUM_CANALES * sizeof(type_for_pixels));
  type_for_pixels *img_padding_2_por_cada_lado_ptr = (type_for_pixels *) malloc((NUM_FILAS+4) * (NUM_COLUMNAS+4) * NUM_CANALES * sizeof(type_for_pixels)); // 4 del padding(2 por cada lado)
  type_for_pixels *img_padding_1_por_cada_lado_ptr = (type_for_pixels *) malloc((NUM_FILAS+2) * (NUM_COLUMNAS+2) * NUM_CANALES * sizeof(type_for_pixels)); // 4 del padding(2 por cada lado)
  type_for_pixels *pool_output_ptr = (type_for_pixels *) malloc(((NUM_FILAS/2)+2) * ((NUM_COLUMNAS/2)+2) * NUM_CANALES * sizeof(type_for_pixels)); // 2 del padding(1 por cada lado) y /2 del stride de 2
  //type_for_pixels *avg_global_ptr = (type_for_pixels *) malloc(NUM_CANALES * sizeof(type_for_pixels)); // 2 del padding(1 por cada lado) y /2 del stride de 2
  //type_for_pixels *conv_S1_output_ptr = (type_for_pixels *) malloc(((NUM_FILAS/2)+2) * ((NUM_COLUMNAS/2)+2) * 2 * sizeof(type_for_pixels)); // Vamos a aplicar 2 filtros
  //type_for_pixels *conv_S2_output_ptr = (type_for_pixels *) malloc(4*4*2* sizeof(type_for_pixels)); // Vamos a aplicar 2 filtros y el resultado es 2x2 pero con los 2 de padding queda 4x4
  type_for_pixels *producto_tensores_ptr = (type_for_pixels *) malloc(2* sizeof(type_for_pixels)); // Solo tenemos dos neuronas


  // CREACION DE LA IMAGEN
  img_ptr = crearImagen(img_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);
  printf("\n======== IMAGEN INICIAL ========");
  visualizarImagen(img_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  // PADDING SAME
  unsigned int PADDING = 1;
  t = clock();
  img_padding_2_por_cada_lado_ptr = add_padding_same_con_copia_imagen(img_ptr, img_padding_2_por_cada_lado_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, PADDING);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  NUM_FILAS += PADDING*2;
  NUM_COLUMNAS += PADDING*2;
  NUM_CANALES += 0;
  STEP_FILA = NUM_COLUMNAS*NUM_CANALES;
  STEP_COLUMNA = NUM_CANALES;

  printf("\n======== IMAGEN INICIAL CON PADDING 2 POR CADA LADO ========\n");
  printf("add_padding_same() tardó %f segundos en ejecutarse \n", time_taken);
  visualizarImagen(img_padding_2_por_cada_lado_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  /*
  // VARIABLES GLOBALES DEL PROGRAMA == CONV 7X7
  unsigned int NUM_FILAS = 3;
  unsigned int NUM_COLUMNAS = 3;
  unsigned int NUM_CANALES = 2;
  unsigned int STEP_FILA = NUM_COLUMNAS*NUM_CANALES;
  unsigned int STEP_COLUMNA = NUM_CANALES;

  double time_taken;
  clock_t t;

  // CREACION DE ESPACIOS DE MEMORIA
  type_for_pixels *img_ptr = (type_for_pixels *) malloc(NUM_FILAS * NUM_COLUMNAS * NUM_CANALES * sizeof(type_for_pixels));
  type_for_pixels *img_padding_3_por_cada_lado_ptr = (type_for_pixels *) malloc((NUM_FILAS+6) * (NUM_COLUMNAS+6) * NUM_CANALES * sizeof(type_for_pixels)); // 6 del padding(3 por cada lado)
  type_for_pixels *conv_7X7_S2_output_ptr = (type_for_pixels *) malloc(4 * 4 * 2 * sizeof(type_for_pixels)); // Vamos a aplicar 2 filtros y de la 3x3 con padding de 1 (5x5) vamos a pasar a 2x2 con padding de 1, es decir 4x4

  // CREACION DE LA IMAGEN
  img_ptr = crearImagen(img_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);
  printf("\n======== IMAGEN INICIAL ========");
  visualizarImagen(img_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  // PADDING SAME
  unsigned int PADDING = 3;
  t = clock();
  img_padding_3_por_cada_lado_ptr = add_padding_same_con_copia_imagen(img_ptr, img_padding_3_por_cada_lado_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, PADDING);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  NUM_FILAS += PADDING*2;
  NUM_COLUMNAS += PADDING*2;
  NUM_CANALES += 0;
  STEP_FILA = NUM_COLUMNAS*NUM_CANALES;
  STEP_COLUMNA = NUM_CANALES;

  printf("\n======== IMAGEN INICIAL CON PADDING 3 POR CADA LADO ========\n");
  printf("add_padding_same() tardó %f segundos en ejecutarse \n", time_taken);
  visualizarImagen(img_padding_3_por_cada_lado_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  // ReLU
  t = clock();
  img_padding_3_por_cada_lado_ptr = ReLU(img_padding_3_por_cada_lado_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, PADDING);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  printf("\n======== ReLU(IMAGEN INICIAL CON PADDING 3 POR CADA LADO) ========\n");
  printf("ReLU() tardó %f segundos en ejecutarse \n", time_taken);
  visualizarImagen(img_padding_3_por_cada_lado_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);
  */


  // Max_Pooling_2x2_S2
  t = clock();
  pool_output_ptr = Max_Pooling_2x2_S2(img_padding_2_por_cada_lado_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, PADDING, 1, pool_output_ptr);
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

  printf("\n======== Max_Pooling_2x2_S2(ReLU(IMAGEN INICIAL CON PADDING 2 POR CADA LADO)) ========\n");
  printf("Max_Pooling_2x2_S2() tardó %f segundos en ejecutarse \n", time_taken);
  visualizarImagen(pool_output_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  /*
  // Avg_Pooling_2x2_S2
  t = clock();
  pool_output_ptr = Avg_Pooling_2x2_S2(img_padding_2_por_cada_lado_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, PADDING, 1, pool_output_ptr);
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

  printf("\n======== Avg_Pooling_2x2_S2(ReLU(IMAGEN INICIAL CON PADDING 2 POR CADA LADO)) ========\n");
  printf("Avg_Pooling_2x2_S2() tardó %f segundos en ejecutarse \n", time_taken);
  visualizarImagen(pool_output_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);
  */

  /*
  // Avg_Pooling_Completo_por_canal
  t = clock();
  avg_global_ptr = Avg_Pooling_Completo_por_canal(img_padding_2_por_cada_lado_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, PADDING, avg_global_ptr);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  NUM_FILAS = 1;
  NUM_COLUMNAS = 1;
  PADDING = 0;
  NUM_FILAS += PADDING*2;
  NUM_COLUMNAS += PADDING*2;
  NUM_CANALES += 0;
  STEP_FILA = NUM_COLUMNAS*NUM_CANALES;
  STEP_COLUMNA = NUM_CANALES;

  printf("\n======== Avg_Pooling_Completo_por_canal(ReLU(IMAGEN INICIAL CON PADDING 2 POR CADA LADO)) ========\n");
  printf("Avg_Pooling_Completo_por_canal() tardó %f segundos en ejecutarse \n", time_taken);
  visualizarImagen(avg_global_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);
  */

  /*
  // Convolucion_3x3_S1
  pool_output_ptr = add_padding_same(pool_output_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, PADDING);
  printf("\n======== add_padding_same(Max_Pooling_2x2_S2(ReLU(IMAGEN INICIAL CON PADDING 2 POR CADA LADO))) ========\n");
  printf("add_padding_same() tardó %f segundos en ejecutarse \n", time_taken);
  visualizarImagen(pool_output_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);

  unsigned int NUM_FILTROS = 2;
  type_for_pixels *kernel = (type_for_pixels *) malloc(3 * 3 * 2 * 2 * sizeof(type_for_pixels));

  *(kernel+0) = (type_for_pixels) 1;
  *(kernel+1) = (type_for_pixels) 2;
  *(kernel+2) = (type_for_pixels) 3;
  *(kernel+3) = (type_for_pixels) 4;
  *(kernel+4) = (type_for_pixels) 5;
  *(kernel+5) = (type_for_pixels) 6;
  *(kernel+6) = (type_for_pixels) 7;
  *(kernel+7) = (type_for_pixels) 8;
  *(kernel+8) = (type_for_pixels) 9;
  *(kernel+9) = (type_for_pixels) 10;
  *(kernel+10) = (type_for_pixels) 11;
  *(kernel+11) = (type_for_pixels) 12;
  *(kernel+12) = (type_for_pixels) 13;
  *(kernel+13) = (type_for_pixels) 14;
  *(kernel+14) = (type_for_pixels) 15;
  *(kernel+15) = (type_for_pixels) 16;
  *(kernel+16) = (type_for_pixels) 17;
  *(kernel+17) = (type_for_pixels) 18;

  *(kernel+18) = (type_for_pixels) 19;
  *(kernel+19) = (type_for_pixels) 20;
  *(kernel+20) = (type_for_pixels) 21;
  *(kernel+21) = (type_for_pixels) 22;
  *(kernel+22) = (type_for_pixels) 23;
  *(kernel+23) = (type_for_pixels) 24;
  *(kernel+24) = (type_for_pixels) 25;
  *(kernel+25) = (type_for_pixels) 26;
  *(kernel+26) = (type_for_pixels) 27;
  *(kernel+27) = (type_for_pixels) 28;
  *(kernel+28) = (type_for_pixels) 29;
  *(kernel+29) = (type_for_pixels) 30;
  *(kernel+30) = (type_for_pixels) 31;
  *(kernel+31) = (type_for_pixels) 32;
  *(kernel+32) = (type_for_pixels) 33;
  *(kernel+33) = (type_for_pixels) 34;
  *(kernel+34) = (type_for_pixels) 35;
  *(kernel+35) = (type_for_pixels) 36;

  t = clock();
  conv3x3_S1_output_ptr = Convolucion_3x3_S1(pool_output_ptr, conv3x3_S1_output_ptr, kernel, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, NUM_FILTROS);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  NUM_CANALES = NUM_FILTROS;
  STEP_FILA = NUM_COLUMNAS*NUM_CANALES;
  STEP_COLUMNA = NUM_CANALES;

  printf("\n======== Convolucion_3x3_S1(Max_Pooling_2x2_S2(ReLU(IMAGEN INICIAL CON PADDING 2 POR CADA LADO))) ========\n");
  printf("Convolucion_3x3_S1() tardó %f segundos en ejecutarse \n", time_taken);
  visualizarImagen(conv3x3_S1_output_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);
  */


  /*
  // Convolucion_3x3_S2
  pool_output_ptr = add_padding_same(pool_output_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, PADDING);
  printf("\n======== add_padding_same(Max_Pooling_2x2_S2(ReLU(IMAGEN INICIAL CON PADDING 2 POR CADA LADO))) ========\n");
  printf("add_padding_same() tardó %f segundos en ejecutarse \n", time_taken);
  visualizarImagen(pool_output_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);

  unsigned int NUM_FILTROS = 2;
  type_for_pixels *kernel = (type_for_pixels *) malloc(3 * 3 * 2 * 2 * sizeof(type_for_pixels));

  *(kernel+0) = (type_for_pixels) 1;
  *(kernel+1) = (type_for_pixels) 2;
  *(kernel+2) = (type_for_pixels) 3;
  *(kernel+3) = (type_for_pixels) 4;
  *(kernel+4) = (type_for_pixels) 5;
  *(kernel+5) = (type_for_pixels) 6;
  *(kernel+6) = (type_for_pixels) 7;
  *(kernel+7) = (type_for_pixels) 8;
  *(kernel+8) = (type_for_pixels) 9;
  *(kernel+9) = (type_for_pixels) 10;
  *(kernel+10) = (type_for_pixels) 11;
  *(kernel+11) = (type_for_pixels) 12;
  *(kernel+12) = (type_for_pixels) 13;
  *(kernel+13) = (type_for_pixels) 14;
  *(kernel+14) = (type_for_pixels) 15;
  *(kernel+15) = (type_for_pixels) 16;
  *(kernel+16) = (type_for_pixels) 17;
  *(kernel+17) = (type_for_pixels) 18;

  *(kernel+18) = (type_for_pixels) 19;
  *(kernel+19) = (type_for_pixels) 20;
  *(kernel+20) = (type_for_pixels) 21;
  *(kernel+21) = (type_for_pixels) 22;
  *(kernel+22) = (type_for_pixels) 23;
  *(kernel+23) = (type_for_pixels) 24;
  *(kernel+24) = (type_for_pixels) 25;
  *(kernel+25) = (type_for_pixels) 26;
  *(kernel+26) = (type_for_pixels) 27;
  *(kernel+27) = (type_for_pixels) 28;
  *(kernel+28) = (type_for_pixels) 29;
  *(kernel+29) = (type_for_pixels) 30;
  *(kernel+30) = (type_for_pixels) 31;
  *(kernel+31) = (type_for_pixels) 32;
  *(kernel+32) = (type_for_pixels) 33;
  *(kernel+33) = (type_for_pixels) 34;
  *(kernel+34) = (type_for_pixels) 35;
  *(kernel+35) = (type_for_pixels) 36;

  unsigned int STEP_FILA_OUTPUT = 4*2;

  t = clock();
  conv_S2_output_ptr = Convolucion_3x3_S2(pool_output_ptr, conv_S2_output_ptr, kernel, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, STEP_FILA_OUTPUT, NUM_FILTROS);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  NUM_FILAS = 4;
  NUM_COLUMNAS = 4;
  NUM_CANALES = NUM_FILTROS;
  STEP_FILA = NUM_COLUMNAS*NUM_CANALES;
  STEP_COLUMNA = NUM_CANALES;
  printf("\n======== Convolucion_3x3_S2(Max_Pooling_2x2_S2(ReLU(IMAGEN INICIAL CON PADDING 2 POR CADA LADO))) ========\n");
  printf("Convolucion_3x3_S2() tardó %f segundos en ejecutarse \n", time_taken);
  visualizarImagen(conv_S2_output_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);
  */


  /*
  // Convolucion_1x1_S1
  pool_output_ptr = add_padding_same(pool_output_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, PADDING);
  printf("\n======== add_padding_same(Max_Pooling_2x2_S2(ReLU(IMAGEN INICIAL CON PADDING 2 POR CADA LADO))) ========\n");
  printf("add_padding_same() tardó %f segundos en ejecutarse \n", time_taken);
  visualizarImagen(pool_output_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);

  unsigned int NUM_FILTROS = 2;
  type_for_pixels *kernel = (type_for_pixels *) malloc(2 * 2 * sizeof(type_for_pixels));

  *(kernel+0) = (type_for_pixels) 1;
  *(kernel+1) = (type_for_pixels) 2;

  *(kernel+2) = (type_for_pixels) 3;
  *(kernel+3) = (type_for_pixels) 4;


  t = clock();
  conv_S1_output_ptr = Convolucion_1x1_S1(pool_output_ptr, conv_S1_output_ptr, kernel, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, NUM_FILTROS);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  NUM_CANALES = NUM_FILTROS;
  STEP_FILA = NUM_COLUMNAS*NUM_CANALES;
  STEP_COLUMNA = NUM_CANALES;

  printf("\n======== Convolucion_1x1_S1(Max_Pooling_2x2_S2(ReLU(IMAGEN INICIAL CON PADDING 2 POR CADA LADO))) ========\n");
  printf("Convolucion_3x3_S1() tardó %f segundos en ejecutarse \n", time_taken);
  visualizarImagen(conv_S1_output_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);
  */

  /*
  // Convolucion_1x1_S2
  pool_output_ptr = add_padding_same(pool_output_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, PADDING);
  printf("\n======== add_padding_same(Max_Pooling_2x2_S2(ReLU(IMAGEN INICIAL CON PADDING 2 POR CADA LADO))) ========\n");
  printf("add_padding_same() tardó %f segundos en ejecutarse \n", time_taken);
  visualizarImagen(pool_output_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);

  unsigned int NUM_FILTROS = 2;
  type_for_pixels *kernel = (type_for_pixels *) malloc(2 * 2 * sizeof(type_for_pixels));

  *(kernel+0) = (type_for_pixels) 1;
  *(kernel+1) = (type_for_pixels) 2;
  *(kernel+2) = (type_for_pixels) 3;
  *(kernel+3) = (type_for_pixels) 4;


  unsigned int STEP_FILA_OUTPUT = 4*2;

  t = clock();
  conv_S2_output_ptr = Convolucion_1x1_S2(pool_output_ptr, conv_S2_output_ptr, kernel, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, STEP_FILA_OUTPUT, NUM_FILTROS);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  NUM_FILAS = 4;
  NUM_COLUMNAS = 4;
  NUM_CANALES = NUM_FILTROS;
  STEP_FILA = NUM_COLUMNAS*NUM_CANALES;
  STEP_COLUMNA = NUM_CANALES;
  printf("\n======== Convolucion_1x1_S2(Max_Pooling_2x2_S2(ReLU(IMAGEN INICIAL CON PADDING 2 POR CADA LADO))) ========\n");
  printf("Convolucion_1x1_S2() tardó %f segundos en ejecutarse \n", time_taken);
  visualizarImagen(conv_S2_output_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);
  */

  /*
  // Convolucion_7x7_S2
  //img_padding_3_por_cada_lado_ptr = add_padding_same(img_padding_3_por_cada_lado_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, PADDING);
  printf("\n======== add_padding_same(ReLU(IMAGEN INICIAL CON PADDING 3 POR CADA LADO)) ========\n");
  printf("add_padding_same() tardó %f segundos en ejecutarse \n", time_taken);
  visualizarImagen(img_padding_3_por_cada_lado_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);

  unsigned int NUM_FILTROS = 2;
  type_for_pixels *kernel = (type_for_pixels *) malloc(7 * 7 * 2 * 2 * sizeof(type_for_pixels));

  for(int i = 0; i<(49*2*2); i++){
    *(kernel+i) = (type_for_pixels) i+1;
  }

  unsigned int STEP_FILA_OUTPUT = 4*NUM_FILTROS; // 4 píxeles en una fila * canales==num_filtros

  t = clock();
  conv_7X7_S2_output_ptr = Convolucion_7x7_S2(img_padding_3_por_cada_lado_ptr, conv_7X7_S2_output_ptr, kernel, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, STEP_FILA_OUTPUT, NUM_FILTROS);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  NUM_FILAS = 4;
  NUM_COLUMNAS = 4;
  NUM_CANALES = NUM_FILTROS;
  STEP_FILA = NUM_COLUMNAS*NUM_CANALES;
  STEP_COLUMNA = NUM_CANALES;
  printf("\n======== Convolucion_7x7_S2(ReLU(IMAGEN INICIAL CON PADDING 3 POR CADA LADO)) ========\n");
  printf("Convolucion_7x7_S2() tardó %f segundos en ejecutarse \n", time_taken);
  visualizarImagen(conv_7X7_S2_output_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);
  */


  /*
  // Suma_dos_tensores elemento a elemento (Residual connection)
  type_for_pixels *matriz = (type_for_pixels *) malloc(6 * 6 * 2 * sizeof(type_for_pixels));
  for(int i = 0; i<(6 * 6 * 2); i++){
    *(matriz+i) = (type_for_pixels) i+1;
  }

  printf("\n======== Matriz ========\n");
  visualizarImagen(matriz, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);

  t = clock();
  pool_output_ptr = suma_elemento_elemento(pool_output_ptr, matriz, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  printf("\n======== suma_elemento_elemento(Max_Pooling_2x2_S2(ReLU(IMAGEN INICIAL CON PADDING 3 POR CADA LADO)), Matriz) ========\n");
  printf("suma_elemento_elemento() tardó %f segundos en ejecutarse \n", time_taken);
  visualizarImagen(pool_output_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);
  */

  /*
  // Suma de vector de bias por canal de la entrada (bias de convoluciones y fully connected)
  type_for_pixels *bias = (type_for_pixels *) malloc(2 * sizeof(type_for_pixels));
  for(int i = 0; i<2; i++){
    *(bias+i) = (type_for_pixels) i+1;
  }

  printf("\n======== Bias ========\n");
  visualizarImagen(bias, 1, 1, NUM_CANALES, 1, 1);

  t = clock();
  pool_output_ptr = suma_vector_con_tensor_por_canales(pool_output_ptr, bias, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  printf("\n======== suma_vector_con_tensor_por_canales(Max_Pooling_2x2_S2(ReLU(IMAGEN INICIAL CON PADDING 3 POR CADA LADO)), Matriz) ========\n");
  printf("suma_vector_con_tensor_por_canales() tardó %f segundos en ejecutarse \n", time_taken);
  visualizarImagen(pool_output_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);
  */

  /*
  // Producto_matricial (fully connected)
  type_for_pixels *weights = (type_for_pixels *) malloc(9*2*2 * sizeof(type_for_pixels));
  for(int i = 0; i<(9*2*2); i++){
    *(weights+i) = (type_for_pixels) i+1;
  }

  printf("\n======== Weights ========\n");
  visualizarImagen(weights, 18, 1, 2, 2, 1); // 18 filas (tam pool_output_ptr en 1D), 1 columna (vector 1D=> 1 columna), 2 canales (2 neuronas) (NO HAY CANALES), step_fila=2, step_columna=1

  t = clock();
  producto_tensores_ptr = producto_tensores(pool_output_ptr, weights, producto_tensores_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA, 2);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds
  printf("\n======== producto_tensores(Max_Pooling_2x2_S2(ReLU(IMAGEN INICIAL CON PADDING 3 POR CADA LADO)), Matriz) ========\n");
  printf("producto_tensores() tardó %f segundos en ejecutarse \n", time_taken);
  visualizarImagen(producto_tensores_ptr, 1, 1, 2, 1, 1);
  */


  // Batch_norm
  type_for_pixels *param_batch_norm = (type_for_pixels *) malloc(2 * sizeof(type_for_pixels));
  for(int i = 0; i<2; i++){
    *(param_batch_norm+i) = (type_for_pixels) i+1;
  }

  printf("\n======== Parametros Batch Norm (media, std) ========\n");
  visualizarImagen(param_batch_norm, 1, 1, NUM_CANALES, 1, 1);

  t = clock();
  pool_output_ptr = batch_norm(pool_output_ptr, param_batch_norm, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);
  t = clock() - t;
  time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  printf("\n======== batch_norm(Max_Pooling_2x2_S2(ReLU(IMAGEN INICIAL CON PADDING 3 POR CADA LADO)), Matriz) ========\n");
  printf("batch_norm() tardó %f segundos en ejecutarse \n", time_taken);
  visualizarImagen(pool_output_ptr, NUM_FILAS, NUM_COLUMNAS, NUM_CANALES, STEP_FILA, STEP_COLUMNA);


  getchar();

  return 0;
}
