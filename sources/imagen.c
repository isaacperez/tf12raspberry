#include "imagen.h"
#include <stdio.h>
#include <stdlib.h>

void visualizarImagen(const type_for_pixels *img_ptr, const unsigned int filas, const unsigned int columnas, const unsigned int canales, const unsigned int STEP_FILA, const unsigned int STEP_COLUMNA){

  /*
  printf("\nÍndices de la imagen...\n");
  for (unsigned int fila = 0; fila < filas; fila++ ) {
    for (unsigned int columna = 0; columna < columnas; columna++ ) {
      for (unsigned int canal = 0; canal < canales; canal++ ) {
        printf("[%i,%i,%i]", fila, columna, canal);
      }
      printf("\t");
    }
    printf("\n");
  }
  */
  printf("\nValores de cada pixel\n");
  for (unsigned int fila = 0; fila < filas; fila++ ) {
    for (unsigned int columna = 0; columna < columnas; columna++ ) {
      for (unsigned int canal = 0; canal < canales; canal++ ) {
        printf("%5.7f ", *(img_ptr + fila*STEP_FILA + columna*STEP_COLUMNA + canal));
      }
      printf(" ");
    }
    printf("\n");
  }

}


type_for_pixels * crearImagen(type_for_pixels *img_ptr, const unsigned int filas, const unsigned int columnas, const unsigned int canales, const unsigned int STEP_FILA, const unsigned int STEP_COLUMNA){

  if (img_ptr){
    printf("Llenamos la imagen...\n");
    for (unsigned int fila = 0; fila < filas; fila++ ) {
      for (unsigned int columna = 0; columna < columnas; columna++ ) {
        for (unsigned int canal = 0; canal < canales; canal++ ) {
          //*(img_ptr + fila*STEP_FILA + columna*STEP_COLUMNA + canal) = (type_for_pixels) (rand() % 100)
          *(img_ptr + fila*STEP_FILA + columna*STEP_COLUMNA + canal) = (type_for_pixels) fila*columnas*canales+columna*canales+canal+1;
        }
      }
    }

  }else{

    printf("No se ha podido reservar memoria para una imagen de %i filas, %i columnas y %i canales.\n", filas, columnas, canales);
    exit(0);

  }

  return img_ptr;
}
