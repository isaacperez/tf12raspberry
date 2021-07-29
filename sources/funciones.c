#include "funciones.h"
#include <stdio.h>

const type_for_pixels cero = (type_for_pixels) 0;
const type_for_pixels cuatro = (type_for_pixels) 4;
const unsigned int STEP_CANAL_DEL_FILTRO_CONV_3x3 = 9;
const unsigned int STEP_CANAL_DEL_FILTRO_CONV_7x7 = 49;


type_for_pixels* add_padding_same_con_copia_imagen(type_for_pixels* img,type_for_pixels* img_padding, const unsigned int FILAS, const unsigned int COLUMNAS, const unsigned int CANALES, const unsigned int STEP_FILA, const unsigned int STEP_COLUMNA, const unsigned int PADDING_SIZE){



  const unsigned int STEP_FILA_IMG_PADDING = (COLUMNAS+PADDING_SIZE*2)*CANALES;
  /*
  const unsigned int ULTIMA_FILA_IMG_PADDING = FILAS+PADDING_SIZE*2;
  const unsigned int ULTIMA_COLUMNA_IMG_PADDING = COLUMNAS+PADDING_SIZE*2;
  const unsigned int ULTIMA_FILA_CON_STEP_FILA = (FILAS-1)*STEP_FILA;
  const unsigned int ULTIMA_COLUMNA_CON_STEP_COLUMNA = (COLUMNAS-1)*STEP_COLUMNA;


  // Llenamos las esquinas
  for(unsigned int fila = 0; fila < PADDING_SIZE; fila++){
    for(unsigned int columna = 0; columna < PADDING_SIZE; columna++){
      for(unsigned int canal = 0; canal < CANALES; canal++){
        *(img_padding+fila*STEP_FILA_IMG_PADDING+columna*STEP_COLUMNA+canal) = *(img+canal);
        *(img_padding+(ULTIMA_FILA_IMG_PADDING-fila-1)*STEP_FILA_IMG_PADDING+columna*STEP_COLUMNA+canal) = *(img+ULTIMA_FILA_CON_STEP_FILA+canal);
        *(img_padding+fila*STEP_FILA_IMG_PADDING+(ULTIMA_COLUMNA_IMG_PADDING-columna-1)*STEP_COLUMNA+canal) = *(img+ULTIMA_COLUMNA_CON_STEP_COLUMNA+canal);
        *(img_padding+(ULTIMA_FILA_IMG_PADDING-fila-1)*STEP_FILA_IMG_PADDING+(ULTIMA_COLUMNA_IMG_PADDING-columna-1)*STEP_COLUMNA+canal) = *(img+ULTIMA_FILA_CON_STEP_FILA+ULTIMA_COLUMNA_CON_STEP_COLUMNA+canal);
      }
    }
  }

  // Llenamos por arriba y abajo
  for (unsigned int fila = 0; fila < PADDING_SIZE; fila++ ) {
    for (unsigned int columna = 0; columna < COLUMNAS; columna++ ) {
      for (unsigned int canal = 0; canal < CANALES; canal++ ) {
        *(img_padding+fila*STEP_FILA_IMG_PADDING+(columna+PADDING_SIZE)*STEP_COLUMNA+canal) = *(img+columna*STEP_COLUMNA+canal);
        *(img_padding+(ULTIMA_FILA_IMG_PADDING-fila-1)*STEP_FILA_IMG_PADDING+(columna+PADDING_SIZE)*STEP_COLUMNA+canal) = *(img+ULTIMA_FILA_CON_STEP_FILA+columna*STEP_COLUMNA+canal);
      }
    }
  }

  // Llenamos por izquierda y por derecha
  for (unsigned int columna = 0; columna < PADDING_SIZE; columna++ ) {
    for (unsigned int fila = 0; fila < FILAS; fila++ ) {
      for (unsigned int canal = 0; canal < CANALES; canal++ ) {
        *(img_padding+(fila+PADDING_SIZE)*STEP_FILA_IMG_PADDING+columna*STEP_COLUMNA+canal) = *(img+fila*STEP_FILA+canal);
        *(img_padding+(fila+PADDING_SIZE)*STEP_FILA_IMG_PADDING+(ULTIMA_COLUMNA_IMG_PADDING-columna-1)*STEP_COLUMNA+canal) = *(img+fila*STEP_FILA+ULTIMA_COLUMNA_CON_STEP_COLUMNA+canal);
      }
    }
  }
  */

  // Copiamos la imagen inicial
  for (unsigned int fila = 0; fila < FILAS; fila++ ) {
    for (unsigned int columna = 0; columna < COLUMNAS; columna++ ) {
      for (unsigned int canal = 0; canal < CANALES; canal++ ) {
        *(img_padding + (fila+PADDING_SIZE)*STEP_FILA_IMG_PADDING + (columna+PADDING_SIZE)*STEP_COLUMNA + canal) = *(img + fila*STEP_FILA + columna*STEP_COLUMNA + canal);
      }
    }
  }


  return img_padding;
}


type_for_pixels* ReLU(type_for_pixels *img,  const unsigned int FILAS, const unsigned int COLUMNAS, const unsigned int CANALES, const unsigned int STEP_FILA, const unsigned int STEP_COLUMNA, const unsigned int PADDING_SIZE){

  // Aplicamos ReLU donde no es PADDING
  const unsigned int limite_FILAS = FILAS - PADDING_SIZE;
  const unsigned int limite_COLUMNAS = COLUMNAS - PADDING_SIZE;

  for (unsigned int fila = PADDING_SIZE; fila < limite_FILAS; fila++ ) {
    for (unsigned int columna = PADDING_SIZE; columna < limite_COLUMNAS; columna++ ) {
      for (unsigned int canal = 0; canal < CANALES; canal++ ) {
        *(img + fila*STEP_FILA + columna*STEP_COLUMNA + canal) = (*(img + fila*STEP_FILA + columna*STEP_COLUMNA + canal) < cero) ? cero : *(img + fila*STEP_FILA + columna*STEP_COLUMNA + canal);
      }
    }
  }
  /*
  for (unsigned int pos = 0; pos < num_pixeles; pos++){
    *(img+pos) = (*(img+pos) < cero) ? cero : *(img+pos);
  }*/

  return img;
}


type_for_pixels* Max_Pooling_2x2_S2(type_for_pixels* img, const unsigned int FILAS, const unsigned int COLUMNAS, const unsigned int CANALES, const unsigned int STEP_FILA, const unsigned int STEP_COLUMNA, const unsigned int STEP_FILA_OUTPUT, const unsigned int PADDING_SIZE_ENTRADA, const unsigned int PADDING_SIZE_SALIDA, type_for_pixels* output){

  unsigned int desp = 0;
  unsigned int desp_output = 0;
  type_for_pixels max;

  // Aplicamos el max_pooling solo a las regiones que no son de padding
  const unsigned int limite_FILAS = FILAS - PADDING_SIZE_ENTRADA;
  const unsigned int limite_COLUMNAS = COLUMNAS - PADDING_SIZE_ENTRADA;

  for (unsigned int fila = PADDING_SIZE_ENTRADA, fila_output = PADDING_SIZE_SALIDA; fila < limite_FILAS; fila+=2, fila_output++) {
    for (unsigned int columna = PADDING_SIZE_ENTRADA, columna_output = PADDING_SIZE_SALIDA; columna < limite_COLUMNAS; columna+=2, columna_output++) {
      for (unsigned int canal = 0; canal < CANALES; canal++ ) {
        desp = fila*STEP_FILA + columna*STEP_COLUMNA + canal;
        desp_output = fila_output*STEP_FILA_OUTPUT + columna_output*STEP_COLUMNA + canal;
        max = (*(img + desp) > *(img + desp + CANALES)) ? *(img + desp) : *(img + desp + CANALES);
        max = (max > *(img + desp + STEP_FILA)) ? max : *(img + desp + STEP_FILA);
        *(output+desp_output) = (max > *(img + desp + CANALES + STEP_FILA)) ? max : *(img + desp + CANALES + STEP_FILA);

        /*if(fila_output==4 && columna_output == 4 && canal == 1){
          printf("%5.5f %5.5f %5.5f %5.5f\n", *(img + desp), *(img + desp + CANALES), *(img + desp + STEP_FILA), *(img + desp + CANALES + STEP_FILA));
          printf("%5.5f\n",*(output+desp_output));
        }*/


      }
    }
  }

  return output;

}


type_for_pixels* Max_Pooling_3x3_S2(type_for_pixels* img, const unsigned int FILAS, const unsigned int COLUMNAS, const unsigned int CANALES, const unsigned int STEP_FILA, const unsigned int STEP_COLUMNA, const unsigned int STEP_FILA_OUTPUT, const unsigned int PADDING_SIZE_ENTRADA, const unsigned int PADDING_SIZE_SALIDA, type_for_pixels* output){

  unsigned int desp = 0;
  unsigned int desp_output = 0;
  type_for_pixels max;

  // Aplicamos el max_pooling solo a las regiones que no son de padding
  const unsigned int limite_FILAS = FILAS - PADDING_SIZE_ENTRADA;
  const unsigned int limite_COLUMNAS = COLUMNAS - PADDING_SIZE_ENTRADA;

  for (unsigned int fila = PADDING_SIZE_ENTRADA, fila_output = PADDING_SIZE_SALIDA; fila < limite_FILAS; fila+=2, fila_output++) {
    for (unsigned int columna = PADDING_SIZE_ENTRADA, columna_output = PADDING_SIZE_SALIDA; columna < limite_COLUMNAS; columna+=2, columna_output++) {
      for (unsigned int canal = 0; canal < CANALES; canal++ ) {
        desp = fila*STEP_FILA + columna*STEP_COLUMNA + canal;
        desp_output = fila_output*STEP_FILA_OUTPUT + columna_output*STEP_COLUMNA + canal;
        // M = max((f,c), (f,c+1))
        max = (*(img + desp) > *(img + desp + CANALES)) ? *(img + desp) : *(img + desp + CANALES);
        // M = max(M, (f,c+2))
        max = (max > *(img + desp + CANALES*2)) ? max : *(img + desp + CANALES*2);
        // M = max(M, (f+1,c))
        max = (max > *(img + desp + STEP_FILA)) ? max : *(img + desp + STEP_FILA);
        // M = max(M, (f+1,c+1))
        max = (max > *(img + desp + STEP_FILA + CANALES)) ? max : *(img + desp + STEP_FILA + CANALES);
        // M = max(M, (f+1,c+2))
        max = (max > *(img + desp + STEP_FILA + CANALES*2)) ? max : *(img + desp + STEP_FILA + CANALES*2);
        // M = max(M, (f+2,c))
        max = (max > *(img + desp + STEP_FILA*2)) ? max : *(img + desp + STEP_FILA*2);
        // M = max(M, (f+2,c+1))
        max = (max > *(img + desp + STEP_FILA*2 + CANALES)) ? max : *(img + desp + STEP_FILA*2 + CANALES);
        // M = max(M, (f+2,c+2))
        *(output+desp_output) = (max > *(img + desp + STEP_FILA*2 + CANALES*2)) ? max : *(img + desp + STEP_FILA*2 + CANALES*2);

        /*if(fila_output==4 && columna_output == 4 && canal == 1){
          printf("%5.5f %5.5f %5.5f %5.5f\n", *(img + desp), *(img + desp + CANALES), *(img + desp + STEP_FILA), *(img + desp + CANALES + STEP_FILA));
          printf("%5.5f\n",*(output+desp_output));
        }*/


      }
    }
  }

  return output;

}


type_for_pixels* Avg_Pooling_2x2_S2(type_for_pixels* img, const unsigned int FILAS, const unsigned int COLUMNAS, const unsigned int CANALES, const unsigned int STEP_FILA, const unsigned int STEP_COLUMNA, const unsigned int STEP_FILA_OUTPUT, const unsigned int PADDING_SIZE_ENTRADA, const unsigned int PADDING_SIZE_SALIDA, type_for_pixels* output){

  /*
  unsigned int desp, desp_output = 0;

  for (unsigned int fila = 0; fila < FILAS; fila+=2 ) {
    for (unsigned int columna = 0; columna < COLUMNAS; columna+=2 ) {
      for (unsigned int canal = 0; canal < CANALES; canal++ ) {
        desp = fila*STEP_FILA + columna*STEP_COLUMNA + canal;
        *(output+desp_output) = (*(img + desp) + *(img + desp + CANALES) + *(img + desp + STEP_FILA) + *(img + desp + STEP_FILA + CANALES)) / cuatro;
        desp_output++;
      }
    }
  }
  */
  unsigned int desp = 0;
  unsigned int desp_output = 0;

  // Aplicamos el max_pooling solo a las regiones que no son de padding
  const unsigned int limite_FILAS = FILAS - PADDING_SIZE_ENTRADA;
  const unsigned int limite_COLUMNAS = COLUMNAS - PADDING_SIZE_ENTRADA;

  for (unsigned int fila = PADDING_SIZE_ENTRADA, fila_output = PADDING_SIZE_SALIDA; fila < limite_FILAS; fila+=2, fila_output++) {
    for (unsigned int columna = PADDING_SIZE_ENTRADA, columna_output = PADDING_SIZE_SALIDA; columna < limite_COLUMNAS; columna+=2, columna_output++) {
      for (unsigned int canal = 0; canal < CANALES; canal++ ) {
        desp = fila*STEP_FILA + columna*STEP_COLUMNA + canal;
        desp_output = fila_output*STEP_FILA_OUTPUT + columna_output*STEP_COLUMNA + canal;
        *(output+desp_output) = (*(img + desp + CANALES + STEP_FILA) + *(img + desp + STEP_FILA) + *(img + desp) + *(img + desp + CANALES)) / cuatro;
      }
    }
  }

  return output;

}


type_for_pixels* Avg_Pooling_Completo_por_canal(type_for_pixels* img, const unsigned int FILAS, const unsigned int COLUMNAS, const unsigned int CANALES, const unsigned int STEP_FILA, const unsigned int STEP_COLUMNA, const unsigned int PADDING_SIZE_ENTRADA, type_for_pixels* output){

  /*
  unsigned int desp_output = 0;
  unsigned int num_pixeles = FILAS*COLUMNAS*CANALES;
  type_for_pixels avg = 0;
  type_for_pixels denominador = FILAS*COLUMNAS;

  for (unsigned int canal = 0; canal < CANALES; canal++ ) {

    avg = 0;
    for (unsigned int i = canal; i < num_pixeles; i+=CANALES ) {
        avg += *(img+i);
    }

    *(output+desp_output) = avg/denominador;
    desp_output++;

  }
  */
  const unsigned int limite_FILAS = FILAS - PADDING_SIZE_ENTRADA;
  const unsigned int limite_COLUMNAS = COLUMNAS - PADDING_SIZE_ENTRADA;

  unsigned int desp_output = 4*CANALES;
  type_for_pixels avg = 0;
  type_for_pixels denominador = (limite_FILAS-PADDING_SIZE_ENTRADA)*(limite_COLUMNAS-PADDING_SIZE_ENTRADA);

  for (unsigned int canal = 0; canal < CANALES; canal++ ) {

    avg = 0;
    for (unsigned int fila = PADDING_SIZE_ENTRADA; fila < limite_FILAS; fila++) {
      for (unsigned int columna = PADDING_SIZE_ENTRADA; columna < limite_COLUMNAS; columna++){
        avg += *(img+fila*STEP_FILA+columna*STEP_COLUMNA+canal);
      }
    }

    *(output+desp_output) = avg/denominador;
    desp_output++;

  }

  return output;

}


type_for_pixels* Convolucion_3x3_S1(type_for_pixels *img, type_for_pixels *img_output, type_for_pixels *kernel, const unsigned int FILAS, const unsigned int COLUMNAS, const unsigned int CANALES, const unsigned int STEP_FILA, const unsigned int STEP_COLUMNA, const unsigned int STEP_FILA_OUTPUT, const unsigned int STEP_COLUMNA_OUTPUT, const unsigned int NUM_FILTROS){

  const unsigned int STEP_FILTRO_3x3 = NUM_FILTROS*STEP_CANAL_DEL_FILTRO_CONV_3x3;
  const unsigned int ultima_fila = FILAS-1;
  const unsigned int ultima_columna = COLUMNAS-1;
  type_for_pixels resultado = (type_for_pixels) 0;

  for(unsigned int filtro = 0; filtro < NUM_FILTROS; filtro++){
    for(unsigned int fila = 1, fila_anterior = 0, fila_posterior = 2; fila < ultima_fila; fila++, fila_anterior++, fila_posterior++){
      for(unsigned int columna = 1, columna_anterior = 0, columna_posterior = 2; columna < ultima_columna; columna++, columna_anterior++, columna_posterior++){
          resultado = 0;
          for(unsigned int canal = 0; canal < CANALES; canal++){

            resultado += *(img+fila*STEP_FILA+columna*STEP_COLUMNA+canal) * *(kernel+filtro+canal*STEP_FILTRO_3x3+4*NUM_FILTROS) +\
                    *(img+fila_anterior*STEP_FILA+columna_anterior*STEP_COLUMNA+canal) * *(kernel+filtro+canal*STEP_FILTRO_3x3) +\
                    *(img+fila_anterior*STEP_FILA+columna*STEP_COLUMNA+canal) * *(kernel+filtro+canal*STEP_FILTRO_3x3+NUM_FILTROS) +\
                    *(img+fila_anterior*STEP_FILA+columna_posterior*STEP_COLUMNA+canal) * *(kernel+filtro+canal*STEP_FILTRO_3x3+2*NUM_FILTROS) +\
                    *(img+fila*STEP_FILA+columna_anterior*STEP_COLUMNA+canal) * *(kernel+filtro+canal*STEP_FILTRO_3x3+3*NUM_FILTROS) +\
                    *(img+fila*STEP_FILA+columna_posterior*STEP_COLUMNA+canal) * *(kernel+filtro+canal*STEP_FILTRO_3x3+5*NUM_FILTROS) +\
                    *(img+fila_posterior*STEP_FILA+columna_anterior*STEP_COLUMNA+canal) * *(kernel+filtro+canal*STEP_FILTRO_3x3+6*NUM_FILTROS) +\
                    *(img+fila_posterior*STEP_FILA+columna*STEP_COLUMNA+canal) * *(kernel+filtro+canal*STEP_FILTRO_3x3+7*NUM_FILTROS) +\
                    *(img+fila_posterior*STEP_FILA+columna_posterior*STEP_COLUMNA+canal) * *(kernel+filtro+canal*STEP_FILTRO_3x3+8*NUM_FILTROS);
            /*
            printf("[%i, %i] Canal %i: ", fila, columna, canal);
            type_for_pixels valor_central = *(img+fila*STEP_FILA+columna*STEP_COLUMNA+canal);
            printf("(%5.2f * %5.2f) + ", valor_central, *(kernel+filtro*STEP_FILTRO_3x3+canal*STEP_CANAL_DEL_FILTRO_CONV_3x3+4));
            type_for_pixels valor_sup_izq = *(img+fila_anterior*STEP_FILA+columna_anterior*STEP_COLUMNA+canal);
            printf("(%5.2f * %5.2f) + ", valor_sup_izq, *(kernel+filtro*STEP_FILTRO_3x3+canal*STEP_CANAL_DEL_FILTRO_CONV_3x3));
            type_for_pixels valor_sup = *(img+fila_anterior*STEP_FILA+columna*STEP_COLUMNA+canal);
            printf("(%5.2f * %5.2f) + ", valor_sup, *(kernel+filtro*STEP_FILTRO_3x3+canal*STEP_CANAL_DEL_FILTRO_CONV_3x3+1));
            type_for_pixels valor_sup_der = *(img+fila_anterior*STEP_FILA+columna_posterior*STEP_COLUMNA+canal);
            printf("(%5.2f * %5.2f) + ", valor_sup_der, *(kernel+filtro*STEP_FILTRO_3x3+canal*STEP_CANAL_DEL_FILTRO_CONV_3x3+2));
            type_for_pixels valor_izq = *(img+fila*STEP_FILA+columna_anterior*STEP_COLUMNA+canal);
            printf("(%5.2f * %5.2f) + ", valor_izq, *(kernel+filtro*STEP_FILTRO_3x3+canal*STEP_CANAL_DEL_FILTRO_CONV_3x3+3));
            type_for_pixels valor_der = *(img+fila*STEP_FILA+columna_posterior*STEP_COLUMNA+canal);
            printf("(%5.2f * %5.2f) + ", valor_der, *(kernel+filtro*STEP_FILTRO_3x3+canal*STEP_CANAL_DEL_FILTRO_CONV_3x3+5));
            type_for_pixels valor_inf_izq = *(img+fila_posterior*STEP_FILA+columna_anterior*STEP_COLUMNA+canal);
            printf("(%5.2f * %5.2f) + ", valor_inf_izq, *(kernel+filtro*STEP_FILTRO_3x3+canal*STEP_CANAL_DEL_FILTRO_CONV_3x3+6));
            type_for_pixels valor_inf = *(img+fila_posterior*STEP_FILA+columna*STEP_COLUMNA+canal);
            printf("(%5.2f * %5.2f) + ", valor_inf, *(kernel+filtro*STEP_FILTRO_3x3+canal*STEP_CANAL_DEL_FILTRO_CONV_3x3+7));

            type_for_pixels valor_inf_der = *(img+fila_posterior*STEP_FILA+columna_posterior*STEP_COLUMNA+canal);
            printf("(%5.2f * %5.2f)\n", valor_inf_der, *(kernel+filtro*STEP_FILTRO_3x3+canal*STEP_CANAL_DEL_FILTRO_CONV_3x3+8));
            */
          }
          *(img_output+fila*STEP_FILA_OUTPUT+columna*STEP_COLUMNA_OUTPUT+filtro) = resultado;
      }
    }
  }

  return img_output;

}

type_for_pixels* Convolucion_3x3_S2(type_for_pixels *img, type_for_pixels *img_output, type_for_pixels *kernel, const unsigned int FILAS, const unsigned int COLUMNAS, const unsigned int CANALES, const unsigned int STEP_FILA, const unsigned int STEP_COLUMNA, const unsigned int STEP_FILA_OUTPUT, const unsigned int STEP_COLUMNA_OUTPUT, const unsigned int NUM_FILTROS){

  const unsigned int STEP_FILTRO_3x3 = NUM_FILTROS*STEP_CANAL_DEL_FILTRO_CONV_3x3;
  const unsigned int ultima_fila = FILAS-1;
  const unsigned int ultima_columna = COLUMNAS-1;
  type_for_pixels resultado = (type_for_pixels) 0;

  for(unsigned int filtro = 0; filtro < NUM_FILTROS; filtro++){
    for(unsigned int fila = 1, fila_1_despues_de_actual = 2, fila_2_despues_de_actual = 3, fila_destino=1; fila < ultima_fila; fila+=2, fila_1_despues_de_actual+=2, fila_2_despues_de_actual+=2, fila_destino++){
      for(unsigned int columna = 1, columna_1_despues_de_actual = 2, columna_2_despues_de_actual = 3, columna_destino=1; columna < ultima_columna; columna+=2, columna_1_despues_de_actual+=2, columna_2_despues_de_actual+=2, columna_destino++){
          resultado = 0;
          for(unsigned int canal = 0; canal < CANALES; canal++){

            resultado += *(img+fila*STEP_FILA+columna*STEP_COLUMNA+canal)* *(kernel+filtro+canal*STEP_FILTRO_3x3) +\
                         *(img+fila*STEP_FILA+columna_1_despues_de_actual*STEP_COLUMNA+canal)* *(kernel+filtro+canal*STEP_FILTRO_3x3+NUM_FILTROS) +\
                         *(img+fila*STEP_FILA+columna_2_despues_de_actual*STEP_COLUMNA+canal)* *(kernel+filtro+canal*STEP_FILTRO_3x3+2*NUM_FILTROS) +\
                         *(img+fila_1_despues_de_actual*STEP_FILA+columna*STEP_COLUMNA+canal)* *(kernel+filtro+canal*STEP_FILTRO_3x3+3*NUM_FILTROS) +\
                         *(img+fila_1_despues_de_actual*STEP_FILA+columna_1_despues_de_actual*STEP_COLUMNA+canal)* *(kernel+filtro+canal*STEP_FILTRO_3x3+4*NUM_FILTROS) +\
                         *(img+fila_1_despues_de_actual*STEP_FILA+columna_2_despues_de_actual*STEP_COLUMNA+canal)* *(kernel+filtro+canal*STEP_FILTRO_3x3+5*NUM_FILTROS) +\
                         *(img+fila_2_despues_de_actual*STEP_FILA+columna*STEP_COLUMNA+canal)* *(kernel+filtro+canal*STEP_FILTRO_3x3+6*NUM_FILTROS) +\
                         *(img+fila_2_despues_de_actual*STEP_FILA+columna_1_despues_de_actual*STEP_COLUMNA+canal)* *(kernel+filtro+canal*STEP_FILTRO_3x3+7*NUM_FILTROS) +\
                         *(img+fila_2_despues_de_actual*STEP_FILA+columna_2_despues_de_actual*STEP_COLUMNA+canal)* *(kernel+filtro+canal*STEP_FILTRO_3x3+8*NUM_FILTROS);

/*
            if(fila==1 && columna==1 && filtro==0){
                printf("Canal de entrada %d: ", canal);
                float a = *(img+fila*STEP_FILA+columna*STEP_COLUMNA+canal)* *(kernel+filtro+canal*STEP_FILTRO_3x3) +\
                             *(img+fila*STEP_FILA+columna_1_despues_de_actual*STEP_COLUMNA+canal)* *(kernel+filtro+canal*STEP_FILTRO_3x3+NUM_FILTROS) +\
                             *(img+fila*STEP_FILA+columna_2_despues_de_actual*STEP_COLUMNA+canal)* *(kernel+filtro+canal*STEP_FILTRO_3x3+2*NUM_FILTROS) +\
                             *(img+fila_1_despues_de_actual*STEP_FILA+columna*STEP_COLUMNA+canal)* *(kernel+filtro+canal*STEP_FILTRO_3x3+3*NUM_FILTROS) +\
                             *(img+fila_1_despues_de_actual*STEP_FILA+columna_1_despues_de_actual*STEP_COLUMNA+canal)* *(kernel+filtro+canal*STEP_FILTRO_3x3+4*NUM_FILTROS) +\
                             *(img+fila_1_despues_de_actual*STEP_FILA+columna_2_despues_de_actual*STEP_COLUMNA+canal)* *(kernel+filtro+canal*STEP_FILTRO_3x3+5*NUM_FILTROS) +\
                             *(img+fila_2_despues_de_actual*STEP_FILA+columna*STEP_COLUMNA+canal)* *(kernel+filtro+canal*STEP_FILTRO_3x3+6*NUM_FILTROS) +\
                             *(img+fila_2_despues_de_actual*STEP_FILA+columna_1_despues_de_actual*STEP_COLUMNA+canal)* *(kernel+filtro+canal*STEP_FILTRO_3x3+7*NUM_FILTROS) +\
                             *(img+fila_2_despues_de_actual*STEP_FILA+columna_2_despues_de_actual*STEP_COLUMNA+canal)* *(kernel+filtro+canal*STEP_FILTRO_3x3+8*NUM_FILTROS);

                printf("%5.7f * %5.7f +", *(img+fila*STEP_FILA+columna*STEP_COLUMNA+canal), *(kernel+filtro+canal*STEP_FILTRO_3x3));
                printf("%5.7f * %5.7f +", *(img+fila*STEP_FILA+columna_1_despues_de_actual*STEP_COLUMNA+canal), *(kernel+filtro+canal*STEP_FILTRO_3x3+NUM_FILTROS));
                printf("%5.7f * %5.7f +", *(img+fila*STEP_FILA+columna_2_despues_de_actual*STEP_COLUMNA+canal), *(kernel+filtro+canal*STEP_FILTRO_3x3+2*NUM_FILTROS));
                printf("%5.7f * %5.7f +", *(img+fila_1_despues_de_actual*STEP_FILA+columna*STEP_COLUMNA+canal), *(kernel+filtro+canal*STEP_FILTRO_3x3+3*NUM_FILTROS));
                printf("%5.7f * %5.7f +", *(img+fila_1_despues_de_actual*STEP_FILA+columna_1_despues_de_actual*STEP_COLUMNA+canal), *(kernel+filtro+canal*STEP_FILTRO_3x3+4*NUM_FILTROS));
                printf("%5.7f * %5.7f +", *(img+fila_1_despues_de_actual*STEP_FILA+columna_2_despues_de_actual*STEP_COLUMNA+canal), *(kernel+filtro+canal*STEP_FILTRO_3x3+5*NUM_FILTROS));
                printf("{%5.7f * %5.7f} +", *(img+fila_2_despues_de_actual*STEP_FILA+columna*STEP_COLUMNA+canal), *(kernel+filtro+canal*STEP_FILTRO_3x3+6*NUM_FILTROS));
                printf("%5.7f * %5.7f +", *(img+fila_2_despues_de_actual*STEP_FILA+columna_1_despues_de_actual*STEP_COLUMNA+canal), *(kernel+filtro+canal*STEP_FILTRO_3x3+7*NUM_FILTROS));
                printf("%5.7f * %5.7f =", *(img+fila_2_despues_de_actual*STEP_FILA+columna_2_despues_de_actual*STEP_COLUMNA+canal), *(kernel+filtro+canal*STEP_FILTRO_3x3+8*NUM_FILTROS));
                printf("%5.7f \n", a);
            }*/
          }
          /*if(fila==1 && columna==1 && filtro==0){
            printf("%5.5f\n", resultado);
          }*/
          *(img_output+fila_destino*STEP_FILA_OUTPUT+columna_destino*STEP_COLUMNA_OUTPUT+filtro) = resultado;
      }
    }
  }

  return img_output;

}


type_for_pixels* Convolucion_1x1_S1(type_for_pixels *img, type_for_pixels *img_output, type_for_pixels *kernel, const unsigned int FILAS, const unsigned int COLUMNAS, const unsigned int CANALES, const unsigned int STEP_FILA, const unsigned int STEP_COLUMNA,  const unsigned int STEP_FILA_OUTPUT, const unsigned int STEP_COLUMNA_OUTPUT, const unsigned int NUM_FILTROS){

  const unsigned int ultima_fila = FILAS-1;
  const unsigned int ultima_columna = COLUMNAS-1;
  type_for_pixels resultado = (type_for_pixels) 0;

  for(unsigned int filtro = 0; filtro < NUM_FILTROS; filtro++){
    for(unsigned int fila = 1; fila < ultima_fila; fila++){
      for(unsigned int columna = 1; columna < ultima_columna; columna++){
          resultado = 0;
          for(unsigned int canal = 0; canal < CANALES; canal++){
            resultado += *(img+fila*STEP_FILA+columna*STEP_COLUMNA+canal) * *(kernel+filtro+canal*NUM_FILTROS);
          }
          *(img_output+fila*STEP_FILA_OUTPUT+columna*STEP_COLUMNA_OUTPUT+filtro) = resultado;
      }
    }
  }

  return img_output;
}


type_for_pixels* Convolucion_1x1_S2(type_for_pixels *img, type_for_pixels *img_output, type_for_pixels *kernel, const unsigned int FILAS, const unsigned int COLUMNAS, const unsigned int CANALES, const unsigned int STEP_FILA, const unsigned int STEP_COLUMNA, const unsigned int STEP_FILA_OUTPUT, const unsigned int STEP_COLUMNA_OUTPUT, const unsigned int NUM_FILTROS){

  const unsigned int ultima_fila = FILAS-1;
  const unsigned int ultima_columna = COLUMNAS-1;
  type_for_pixels resultado = (type_for_pixels) 0;

  for(unsigned int filtro = 0; filtro < NUM_FILTROS; filtro++){
    for(unsigned int fila = 1, fila_destino=1; fila < ultima_fila; fila+=2, fila_destino++){
      for(unsigned int columna = 1, columna_destino=1; columna < ultima_columna; columna+=2, columna_destino++){
          resultado = 0;
          for(unsigned int canal = 0; canal < CANALES; canal++){
            resultado += *(img+fila*STEP_FILA+columna*STEP_COLUMNA+canal) * *(kernel+filtro+canal*NUM_FILTROS);
          }
          *(img_output+fila_destino*STEP_FILA_OUTPUT+columna_destino*STEP_COLUMNA_OUTPUT+filtro) = resultado;
      }
    }
  }

  return img_output;

}


type_for_pixels* Convolucion_7x7_S2(type_for_pixels *img, type_for_pixels *img_output, type_for_pixels *kernel, const unsigned int FILAS, const unsigned int COLUMNAS, const unsigned int CANALES, const unsigned int STEP_FILA, const unsigned int STEP_COLUMNA, const unsigned int STEP_FILA_OUTPUT, const unsigned int NUM_FILTROS){

  const unsigned int ultima_fila = FILAS-6; // 2 de padding por cada lado
  const unsigned int ultima_columna = COLUMNAS-6; // 2 de padding por cada lado
  const unsigned int STEP_FILTRO_7x7 = NUM_FILTROS*STEP_CANAL_DEL_FILTRO_CONV_7x7;
  const unsigned int STEP_FILA_FILTRO_7X7 = 7;
  type_for_pixels resultado = (type_for_pixels) 0;

  // ATENCION: TENSORFLOW CONVOLUCIONA EN CADA POSICIÓN (X,Y) DE LA IMAGEN PERO SI SE EMPLEA STRIDE, SE CONVOLUCIONA EN LA POSICION (X+STRIDE, Y+STRIDE) DANDO SALTOS DE STRIDE EN STRIDE EN LAS DIMENSIONES DE LA ENTRADA
  // LA CONVOLUCION SE REALIZA PONIENDO EL CENTRO DEL KERNEL SOBRE EL PUNTO (X+STRIDE, Y+STRIDE) => LA ESQUINA SUPERIOR IZQUIERDA DE LA IMAGEN CON LA POSICION (0,0) DEL kernel
  // COMO LA ENTRADA LLEVA 3 DE PADDING, EL PUNTO SUPERIOR IZQUIERDO A CONVOLUCIONAR ES EL (1,1)

  for(unsigned int fila = 1, fila_destino=1; fila < ultima_fila; fila+=2, fila_destino++){
    for(unsigned int columna = 1, columna_destino=1; columna < ultima_columna; columna+=2, columna_destino++){
      //printf("[%d,%d]\n", fila_destino, columna_destino);
      for(unsigned int filtro = 0; filtro < NUM_FILTROS; filtro++){
        //printf("Filtro %i\n", filtro);
        resultado = 0;
        for(unsigned int canal = 0; canal < CANALES; canal++){
          //printf("Canal %i\n", canal);
        //  if(fila==4){
            //printf("(%i, %i, %i): %5.2f\n", fila, columna, canal, *(img+(fila)*STEP_FILA+(columna)*STEP_COLUMNA+canal));
        //  }
          // Recorremos el canal correspondiente del filtro y lo aplicamos a la entrada en la posición actual
          for(unsigned int fila_actual = 0; fila_actual < 7; fila_actual++){
            for(unsigned int columna_actual = 0; columna_actual < 7; columna_actual++){
              //if (fila == 2){
              //  printf("(%i, %i): %5.2f %5.2f\n", fila_actual, columna_actual, *(img+(fila+fila_actual)*STEP_FILA+(columna+columna_actual)*STEP_COLUMNA+canal), *(kernel+filtro*STEP_FILTRO_7x7+canal*STEP_CANAL_DEL_FILTRO_CONV_7x7+fila_actual*STEP_FILA_FILTRO_7X7+columna_actual));
              //}
              //if(fila==4 && columna == 2){
              //  printf("(%i, %i): %5.7f %5.7f (%i, %i, %i)\n", fila_actual, columna_actual, *(img+(fila+fila_actual)*STEP_FILA+(columna+columna_actual)*STEP_COLUMNA+canal),*(kernel+canal*STEP_FILTRO_7x7+fila_actual*(STEP_FILA_FILTRO_7X7*NUM_FILTROS)+columna_actual*NUM_FILTROS+filtro), fila_actual*(STEP_FILA_FILTRO_7X7*NUM_FILTROS), columna_actual*NUM_FILTROS, filtro);

              //}
              //if (fila==0 && columna == 0){
                //printf("[%d, %d, %d, %d]: %5.5f, %5.5f\n", fila_actual, columna_actual, canal, filtro, *(img+(fila+fila_actual)*STEP_FILA+(columna+columna_actual)*STEP_COLUMNA+canal), *(kernel+canal*STEP_FILTRO_7x7+fila_actual*(STEP_FILA_FILTRO_7X7*NUM_FILTROS)+columna_actual*NUM_FILTROS+filtro));
              //  printf("[%d]: %5.5f * %5.5f+...\n",canal*STEP_FILTRO_7x7+fila_actual*(STEP_FILA_FILTRO_7X7*NUM_FILTROS)+columna_actual*NUM_FILTROS+filtro, *(img+(fila+fila_actual)*STEP_FILA+(columna+columna_actual)*STEP_COLUMNA+canal), *(kernel+canal*STEP_FILTRO_7x7+fila_actual*(STEP_FILA_FILTRO_7X7*NUM_FILTROS)+columna_actual*NUM_FILTROS+filtro));
              //}
              resultado += *(img+(fila+fila_actual)*STEP_FILA+(columna+columna_actual)*STEP_COLUMNA+canal) * *(kernel+canal*STEP_FILTRO_7x7+fila_actual*(STEP_FILA_FILTRO_7X7*NUM_FILTROS)+columna_actual*NUM_FILTROS+filtro);
            }
          }

        }

        //if(fila==1 && columna == 1){
        //  printf(": %5.7f, %5i, %5i, %5i, %5i, %5i\n", resultado, fila_destino, STEP_FILA_OUTPUT, columna_destino, NUM_FILTROS, filtro);
        //}
        *(img_output+fila_destino*STEP_FILA_OUTPUT+columna_destino*NUM_FILTROS+filtro) = resultado;
      }
    }
  }

  return img_output;

}


type_for_pixels* add_padding_same(type_for_pixels* img, const unsigned int FILAS, const unsigned int COLUMNAS, const unsigned int CANALES, const unsigned int STEP_FILA, const unsigned int STEP_COLUMNA, const unsigned int PADDING_SIZE){

  // Llenamos las esquinas
  const unsigned int FILAS_con_uno_menos = FILAS -1;
  const unsigned int COLUMNAS_con_uno_menos = COLUMNAS -1;
  for(unsigned int fila = 0; fila < PADDING_SIZE; fila++){
    for(unsigned int columna = 0; columna < PADDING_SIZE; columna++){
      for(unsigned int canal = 0; canal < CANALES; canal++){
        *(img+fila*STEP_FILA+columna*STEP_COLUMNA+canal) = (type_for_pixels) 0.0;//*(img+PADDING_SIZE*STEP_FILA+PADDING_SIZE*STEP_COLUMNA+canal);
        *(img+(FILAS_con_uno_menos-fila)*STEP_FILA+columna*STEP_COLUMNA+canal) = (type_for_pixels) 0.0;//*(img+(FILAS-PADDING_SIZE -1)*STEP_FILA+PADDING_SIZE*STEP_COLUMNA+canal);
        *(img+fila*STEP_FILA+(COLUMNAS_con_uno_menos-columna)*STEP_COLUMNA+canal) = (type_for_pixels) 0.0;//*(img+PADDING_SIZE*STEP_FILA+(COLUMNAS-PADDING_SIZE -1)*STEP_COLUMNA+canal);
        *(img+(FILAS_con_uno_menos-fila)*STEP_FILA+(COLUMNAS_con_uno_menos-columna)*STEP_COLUMNA+canal) = (type_for_pixels) 0.0;//*(img+(FILAS-PADDING_SIZE -1)*STEP_FILA+(COLUMNAS-PADDING_SIZE -1)*STEP_COLUMNA+canal);
      }
    }
  }

  // Llenamos por arriba y abajo
  const unsigned int ultima_columna = COLUMNAS-PADDING_SIZE;
  for (unsigned int fila = 0; fila < PADDING_SIZE; fila++ ) {
    for (unsigned int columna = PADDING_SIZE; columna < ultima_columna; columna++ ) {
      for (unsigned int canal = 0; canal < CANALES; canal++ ) {
        //printf("%5.2f\n", *(img+PADDING_SIZE*STEP_FILA+columna*STEP_COLUMNA+canal));
        *(img+fila*STEP_FILA+columna*STEP_COLUMNA+canal) = (type_for_pixels) 0.0;//*(img+PADDING_SIZE*STEP_FILA+columna*STEP_COLUMNA+canal);
        *(img+(FILAS_con_uno_menos-fila)*STEP_FILA+columna*STEP_COLUMNA+canal) = (type_for_pixels) 0.0;//*(img+(FILAS_con_uno_menos-PADDING_SIZE)*STEP_FILA+columna*STEP_COLUMNA+canal);
      }
    }
  }

  // Llenamos por izquierda y por derecha
  const unsigned int ultima_fila = FILAS-PADDING_SIZE;
  for (unsigned int columna = 0; columna < PADDING_SIZE; columna++ ) {
    for (unsigned int fila = PADDING_SIZE; fila < ultima_fila; fila++ ) {
      for (unsigned int canal = 0; canal < CANALES; canal++ ) {
        *(img+fila*STEP_FILA+columna*STEP_COLUMNA+canal) = (type_for_pixels) 0.0;//*(img+fila*STEP_FILA+PADDING_SIZE*STEP_COLUMNA+canal);
        *(img+fila*STEP_FILA+(COLUMNAS_con_uno_menos-columna)*STEP_COLUMNA+canal) = (type_for_pixels) 0.0;//*(img+fila*STEP_FILA+(COLUMNAS_con_uno_menos-PADDING_SIZE)*STEP_COLUMNA+canal);
      }
    }
  }

  return img;

}


type_for_pixels* suma_elemento_elemento(type_for_pixels* img1, const type_for_pixels* img2, const unsigned int FILAS, const unsigned int COLUMNAS, const unsigned int CANALES, const unsigned int STEP_FILA, const unsigned int STEP_COLUMNA){
  // Se entiende que img1 tiene un padding de 1 y que es donde se va a alojar el resultado de la suma de img1 + img2

  const unsigned int limite_FILAS = FILAS - 1;
  const unsigned int limite_COLUMNAS = COLUMNAS - 1;

  for (unsigned int fila = 1; fila < limite_FILAS; fila++) {
    for (unsigned int columna = 1; columna < limite_COLUMNAS; columna++) {
      for (unsigned int canal = 0; canal < CANALES; canal++ ) {
        *(img1+fila*STEP_FILA + columna*STEP_COLUMNA + canal) += *(img2+fila*STEP_FILA + columna*STEP_COLUMNA + canal);
      }
    }
  }

  return img1;
}


type_for_pixels* suma_vector_con_tensor_por_canales(type_for_pixels* img, const type_for_pixels* vector, const unsigned int FILAS, const unsigned int COLUMNAS, const unsigned int CANALES, const unsigned int STEP_FILA, const unsigned int STEP_COLUMNA,const unsigned int tiene_padding_la_entrada){
  // Se entiende que img tiene un padding de 1 y que es donde se va a alojar el resultado de la suma de img1 + img2

  if(tiene_padding_la_entrada>0){
    const unsigned int limite_FILAS = FILAS - 1;
    const unsigned int limite_COLUMNAS = COLUMNAS - 1;
    type_for_pixels bias_actual = (type_for_pixels) 0;

    for (unsigned int canal = 0; canal < CANALES; canal++ ) {
      bias_actual = *(vector+canal);

      for (unsigned int fila = 1; fila < limite_FILAS; fila++) {
        for (unsigned int columna = 1; columna < limite_COLUMNAS; columna++) {
          *(img+fila*STEP_FILA + columna*STEP_COLUMNA + canal) += bias_actual;
        }
      }
    }
  }
  else{
    const unsigned int limite_FILAS = FILAS;
    const unsigned int limite_COLUMNAS = COLUMNAS;
    type_for_pixels bias_actual = (type_for_pixels) 0;

    for (unsigned int canal = 0; canal < CANALES; canal++ ) {
      bias_actual = *(vector+canal);
      for (unsigned int fila = 0; fila < limite_FILAS; fila++) {
        for (unsigned int columna = 0; columna < limite_COLUMNAS; columna++) {
          *(img+fila*STEP_FILA + columna*STEP_COLUMNA + canal) += bias_actual;
        }
      }
    }
  }
  return img;

}

type_for_pixels* producto_tensores(const type_for_pixels* img, const type_for_pixels* weights, type_for_pixels* producto_tensores, const unsigned int FILAS, const unsigned int COLUMNAS, const unsigned int CANALES, const unsigned int STEP_FILA, const unsigned int STEP_COLUMNA, const unsigned int NUM_NEURONAS){
  // Se entiende que img tiene un padding de 1 y que weights tiene la forma de (img en 1D)x#_NEURONAS y producto_tensores tiene la forma #_NEURONAS

  const unsigned int limite_FILAS = FILAS - 1;
  const unsigned int limite_COLUMNAS = COLUMNAS - 1;
  const unsigned int TAM_INPUT = (FILAS-2) * (COLUMNAS-2) * CANALES;

  unsigned int indice_vector_neurona = 0;
  type_for_pixels suma = (type_for_pixels) 0;

  for (unsigned int neurona = 0; neurona < NUM_NEURONAS; neurona++ ){
    indice_vector_neurona = 0;
    suma = 0;
    //sprintf("Neurona %i\n", neurona);
    for (unsigned int fila = 1; fila < limite_FILAS; fila++) {
      for (unsigned int columna = 1; columna < limite_COLUMNAS; columna++) {
        for (unsigned int canal = 0; canal < CANALES; canal++ ) {
          //printf("%5.2f * %5.2f\n", *(img+fila*STEP_FILA + columna*STEP_COLUMNA + canal), *(weights+neurona+indice_vector_neurona));
          suma += *(img+fila*STEP_FILA + columna*STEP_COLUMNA + canal) * *(weights+neurona*TAM_INPUT+indice_vector_neurona);
          //printf("%5.2f ", *(weights+neurona*TAM_INPUT+indice_vector_neurona));
          indice_vector_neurona++;
        }
      }
    }
    //printf("\n\n");
    *(producto_tensores+neurona) = suma;
  }

  return producto_tensores;

}


type_for_pixels* batch_norm(type_for_pixels* img, const type_for_pixels* param_batch_norm, const unsigned int FILAS, const unsigned int COLUMNAS, const unsigned int CANALES, const unsigned int STEP_FILA, const unsigned int STEP_COLUMNA){
  // Se entiende que img tiene un padding de 1 y que es donde se va a alojar el resultado de aplicar batch normalization con media como primer elemento de param_batch_norm y std como segundo

  const unsigned int limite_FILAS = FILAS - 1;
  const unsigned int limite_COLUMNAS = COLUMNAS - 1;
  type_for_pixels media = (type_for_pixels) *param_batch_norm;
  type_for_pixels std = (type_for_pixels) *(param_batch_norm+1);

  for (unsigned int fila = 1; fila < limite_FILAS; fila++) {
    for (unsigned int columna = 1; columna < limite_COLUMNAS; columna++) {
      for (unsigned int canal = 0; canal < CANALES; canal++ ) {
        *(img+fila*STEP_FILA + columna*STEP_COLUMNA + canal) -= std;
        *(img+fila*STEP_FILA + columna*STEP_COLUMNA + canal) /= media;
      }
    }
  }

  return img;

}
