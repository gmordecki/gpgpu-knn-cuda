#include <stdio.h>
#include <stdlib.h>
#include <string.h>


typedef struct DataPoint {
    int id;
    int tag;
    float *vector;
    struct DataPoint *next;
} DataPoint;


DataPoint* Create_DataPoint(){
    DataPoint *dp=(DataPoint*)malloc(sizeof(DataPoint));
    dp->id = -1;
    dp->tag = -1;//Nothing at first
    dp->vector = NULL;//Nothing at first
    dp->next = NULL;//Nothing at first
    return dp;
}


void DeleteAllDataPoint(DataPoint *dp){
    if(dp->next){
        DeleteAllDataPoint(dp->next);
        dp->next=NULL;
    }else{
        free(dp->vector);
        free(dp);
    }
}


int readCSV(const char *file, DataPoint *param_dp, int dimension, int presicion)
{
    // dimension + 1 por el tag
    // presicion+3 por el primer caracter, la coma y el punto
    // +8 para estar seguro. Tenía que agregar algo por el \n
    int BSIZE = (dimension + 1) * sizeof(char) * (presicion + 3) + 8;

    char buffer[BSIZE];
    FILE *f;
    char *field;
    DataPoint* dp=param_dp;
    // open the CSV file
    f = fopen(file,"r");
    if( f == NULL)
    {
        printf("Unable to open file '%s'\n",file);
        exit(1);
    }

    // read the data
    int lines = 0;
    while(fgets(buffer,BSIZE,f) != NULL)
    {
        if(dp->next==NULL){
            dp->next=Create_DataPoint();
        }
        dp=dp->next; //dp is New dp now

        dp->id = lines;

        field = strtok(buffer,",");

        dp->tag = atof(field);

        int col_index = 0;
        dp->vector=(float *)malloc(dimension * sizeof(float));
        for (field = strtok(NULL, ","); field != NULL && col_index < dimension; field = strtok(NULL, ",")) {
            dp->vector[col_index++] = atof(field);
        }

        lines++;
    }
    // close file
    fclose(f);
    return lines;
}
