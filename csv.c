#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define MAXROWS 10
#define DIMENSION 4
#define BSIZE 128

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


// int load_dataset(int max_rows, int max_cols, char* path, float* rows) {

//   int row_index = 0;
//   char line[128];
//   char* token = NULL;
//   FILE* fp = fopen(path,"r");
//   char buf[max_cols], *p = buf;
//   size_t cols = 1;
//   if (fp != NULL)
//   {
//     if (!fgets (buf, max_cols, fp)) {                       /* read / validate headings row */
//         fputs ("error: empty file.\n", stderr);
//         return 1;
//     }

//     while (*p && (p = strchr (p, ','))) {               /* loop counting ',' */
//         cols++;
//         p++;
//     }
//     printf("cols es %ld y p es %s", cols, p);

//     while (fgets( line, sizeof(line), fp) != NULL && row_index < max_rows)
//     {
//       int col_index = 0;
//       for (token = strtok(line, ","); token != NULL && col_index < cols; token = strtok(NULL, ","))
//       {
//         rows[row_index][col_index++] = atof(token);
//       }
//       row_index++;
//     }
//     fclose(fp);
//   }

//   return row_index;
// }

int readCSV(const char *file, DataPoint *param_dp)
{
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
        if(dp->next==NULL){//Next Nothing So Create after it
            dp->next=Create_DataPoint();
        }
        dp=dp->next;//dp is New dp now

        dp->id = lines;

        field = strtok(buffer,",");
        dp->tag = atof(field);

        int col_index = 0;
        dp->vector=(float *)malloc(DIMENSION * sizeof(float));
        for (field = strtok(NULL, ","); field != NULL && col_index < DIMENSION; field = strtok(NULL, ",")) {
            dp->vector[col_index++] = atof(field);
        }

        // // get value
        // field = strtok(NULL,",");
        // //Q_Q
        //     //<--------Here alloc memory for your value because strtok not alloc new memory it just return a pointer in buffer[?]-------------->
        //     dp->value=(char *)malloc((sizeof(strlen(field)+1)*sizeof(char)));//+1 Becuz '\0' at end of string is necessary
        //     //<--------Here Copy Result-------------->
        //     strcpy(dp->value, field);
        // //Q_Q

        // display the result
        lines++;
    }
    //close file
    fclose(f);
    return lines;
}
