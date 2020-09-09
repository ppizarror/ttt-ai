#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include "Tinn.h"

#include <CL/cl.h>

#define	LOCAL_SIZE 150

typedef struct ans{
    int id;
    int fitness;
    Tinn red;
} ans;

const char* CL_FILE_NAME = {"ttt_kernel.cl"};
const char* save_file = "mejor_red";
int iter = 1000;

void Wait(cl_command_queue);
void print_ans(ans* respuestas, int l);
bool compare_ans(ans a, ans b);
Tinn* duplicate(Tinn* red, int size);
Tinn* mutation(Tinn* red, int size);
Tinn* crossover(Tinn* ann, ans* resultados, int size, int &new_size);
static float frand();
static void from(const Tinn t, float* pesos);
Tinn xtbuildfrom(const int nips, const int nhid, const int nops, float* pesos);
void Wait(cl_command_queue queue);


int main(int argc, char* argv[]){
    
    //setting devices
    cl_uint numPlatforms;
    cl_int status = clGetPlatformIDs(0, NULL, &numPlatforms);
    if(status != CL_SUCCESS) fprintf(stderr, "clGetPlatformIDs failed (1)\n");
    
    //fprintf(stderr, "Number of platforms = %d\n", numPlatforms);
    cl_platform_id* platforms = new cl_platform_id[numPlatforms];
    status = clGetPlatformIDs(numPlatforms, platforms, NULL);
    if(status != CL_SUCCESS) fprintf(stderr, "clGetPlatformIDs failed (2)\n");
    cl_device_id device;
    status = clGetDeviceIDs(*platforms, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    //memory buffers
    int pop_amount = LOCAL_SIZE;
    srand (time(NULL));

    char h_boards[pop_amount][3][3];
    for(int k=0;k<pop_amount;k++){
        for(int i=0;i<3;i++){
            for(int j=0;j<3; j++){
                h_boards[k][i][j] = '_';
            }
        }
    }

    Tinn h_ann[pop_amount];
    for(int i=0; i<pop_amount; i++){
        h_ann[i] = xtbuild(27, 243, 9);

    }

    int h_random[pop_amount][20];
    for(int i=0;i<pop_amount;i++){
        for(int j=0; j<20;j++){
            h_random[i][j] = (int)rand()%9;
        }
    }


    ans h_respuestas[pop_amount];

    size_t dataSize_boards = pop_amount*3*3*sizeof(char);
    size_t dataSize_ann = sizeof(Tinn)*pop_amount;
    size_t dataSize_respuestas = sizeof(ans)*pop_amount;
    size_t dataSize_random = sizeof(int)*pop_amount*20;

    //create a context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
    if(status != CL_SUCCESS) fprintf(stderr, "clCreateContext failed\n");

    //create a command queue
    cl_command_queue cmdQueue = clCreateCommandQueue(context, device, 0, &status);
    if(status != CL_SUCCESS) fprintf(stderr, "clCreateCommandQueue failed\n");

    // allocate memory buffers on  the device
    cl_mem d_ann = clCreateBuffer(context, CL_MEM_READ_WRITE, dataSize_ann, NULL, &status);
    if(status != CL_SUCCESS) fprintf(stderr, "clCreateBuffer failed (d_ann)\n");
    cl_mem d_boards = clCreateBuffer(context, CL_MEM_READ_WRITE, dataSize_boards, NULL, &status);
    if(status != CL_SUCCESS) fprintf(stderr, "clCreateBuffer failed (d_boards)\n");
    cl_mem d_respuestas = clCreateBuffer(context, CL_MEM_READ_WRITE, dataSize_respuestas, NULL, &status);
    if(status != CL_SUCCESS) fprintf(stderr, "clCreateBuffer failed (d_respuestas)\n");
    cl_mem d_random = clCreateBuffer(context, CL_MEM_READ_ONLY, dataSize_random, NULL, &status);
    if(status != CL_SUCCESS) fprintf(stderr, "clCreateBuffer failed (d_random)\n");

    
    
    //enqueue the buffers
    status = clEnqueueWriteBuffer(cmdQueue, d_ann, CL_FALSE, 0, dataSize_ann, h_ann, 0, NULL, NULL);
    if(status != CL_SUCCESS) fprintf(stderr, "clEnqueueWriteBuffer failed (h_ann -> d_ann)\n");

    status = clEnqueueWriteBuffer(cmdQueue, d_boards, CL_FALSE, 0, dataSize_boards, h_boards, 0, NULL, NULL);
    if(status != CL_SUCCESS) fprintf(stderr, "clEnqueueWriteBuffer failed (h_boards -> d_boards)\n");

    status = clEnqueueWriteBuffer(cmdQueue, d_respuestas, CL_FALSE, 0, dataSize_respuestas, h_respuestas, 0, NULL, NULL);
    if(status != CL_SUCCESS) fprintf(stderr, "clEnqueueWriteBuffer failed (h_respuestas -> d_respuestas)\n");

    status = clEnqueueWriteBuffer(cmdQueue, d_random, CL_FALSE, 0, dataSize_random, h_random, 0, NULL, NULL);
    if(status != CL_SUCCESS) fprintf(stderr, "clEnqueueWriteBuffer failed (h_random -> d_random)\n");



    //open de cl content
    FILE* fp = fopen(CL_FILE_NAME, "r");
    if(fp == NULL)
    {
        fprintf(stderr, "Cannot open OpenCL source\n");
        return 1;
    }

    //read the file
    fseek(fp, 0, SEEK_END);
    size_t fileSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char* clProgramText = new char[fileSize+1];
    size_t n = fread(clProgramText, 1, fileSize, fp);
    clProgramText[fileSize] = '\0';
    fclose(fp);

    //create the kernel in the device
    char* strings[1];
    strings[0] = clProgramText;
    cl_program program = clCreateProgramWithSource(context, 1, (const char **) strings, NULL, &status);
    if(status != CL_SUCCESS) fprintf(stderr, "clCreateProgramWithSource failed\n");
    delete[] clProgramText;

    //build the kernel on device
    char* options = {(char*)""};
    status = clBuildProgram(program, 1, &device, options, NULL, NULL);
    if(status != CL_SUCCESS)
    {
        size_t size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &size);
        cl_char* log = new cl_char[size];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, size, log, NULL);
        fprintf(stderr, "clGetProgramBuildInfo failed:\n%s\n", log);
        delete[] log;
    }

    //create kernel object
    cl_kernel kernel = clCreateKernel(program, "match", &status);
    if(status != CL_SUCCESS) fprintf(stderr, "clCreateKernel failed\n");
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_ann);
    if(status != CL_SUCCESS) fprintf(stderr, "clSetKernelArg failed (d_ann)\n");
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_respuestas);
    if(status != CL_SUCCESS) fprintf(stderr, "clSetKernelArg failed (d_respuestas)\n");
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_boards);
    if(status != CL_SUCCESS) fprintf(stderr, "clSetKernelArg failed (d_boards)\n");
    status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_random);
    if(status != CL_SUCCESS) fprintf(stderr, "clSetKernelArg failed (d_random)\n");


    //Enqueue the kernel for execution
    //size_t linGlobal = (size_t)ceil(w/(float)LOCAL_SIZE)*LOCAL_SIZE;
    size_t globalWorkSize[2] = {LOCAL_SIZE, 1};
    size_t localWorkSize[2] = {LOCAL_SIZE, 1};

    Wait(cmdQueue);
    
    //start loop
    for(int it= 0; it<iter; it++)
    {
        std::cout << "Generacion: " << it+1 << std::endl;

        status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
        if(status != CL_SUCCESS) fprintf(stderr, "clEnqueueNDRangeKernel failed\n");

        status = clEnqueueReadBuffer(cmdQueue, d_respuestas, CL_TRUE, 0, dataSize_respuestas, h_respuestas, 0, NULL, NULL);
        if(status != CL_SUCCESS) fprintf(stderr, "clEnqueueReadBuffer d_respuestas failed\n");

        status = clEnqueueReadBuffer(cmdQueue, d_ann, CL_TRUE, 0, dataSize_ann, h_ann, 0, NULL, NULL);
        if(status != CL_SUCCESS) fprintf(stderr, "clEnqueueReadBuffer d_ann failed\n");

        Wait(cmdQueue);

        std::sort(h_respuestas, h_respuestas + pop_amount, compare_ans);
        
        int size;

        Tinn* crossovered_net = crossover(h_ann, h_respuestas, pop_amount, size);

        int fixed_size = size*2;
        
        Tinn* duplicated_net = duplicate(crossovered_net, fixed_size);
    
        Tinn* next_gen_net = mutation(duplicated_net, fixed_size);

        status = clEnqueueWriteBuffer(cmdQueue, d_ann, CL_FALSE, 0, dataSize_ann, next_gen_net, 0, NULL, NULL);
        if(status != CL_SUCCESS) fprintf(stderr, "clEnqueueWriteBuffer failed (h_ann -> d_ann)\n");

        status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_ann);
        if(status != CL_SUCCESS) fprintf(stderr, "clSetKernelArg failed (d_ann)\n");

        if(it==iter-1)
        {
            xtsave(next_gen_net[0], save_file);
        }

    }

    Wait(cmdQueue);

    clFinish(cmdQueue);

    print_ans(h_respuestas, pop_amount);

    //cleaning up
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdQueue);
    clReleaseMemObject(d_ann);
    clReleaseMemObject(d_boards);
    clReleaseMemObject(d_respuestas);
    clReleaseMemObject(d_random);

    return 0;
}

void print_ans(ans* respuestas, int l)
{
    for(int i=0; i<l; i++)
    {
        fprintf(stderr, "(%d,%d) ", respuestas[i].id, respuestas[i].fitness);
    }
    std::cout << std::endl;
}

bool compare_ans(ans a, ans b) 
{ 
    return (a.fitness > b.fitness); 
}

Tinn* crossover(Tinn* ann, ans* resultados, int size, int &new_size)
{
    srand (time(NULL));

    Tinn* crossovered_net = (Tinn*)calloc(size/2, sizeof(Tinn)); // Nuevo arreglo de redes neuronales
    int num_pesos = resultados->red.nw;
    int cont = 0;

    for(int i=0; i<size/2; i++) // recorre los n/2 mejores resultados
    {
        int id_red_1 = resultados[i].id;
        int id_red_2 = resultados[i+1].id;
        Tinn red_neuronal_1 = ann[id_red_1];
        Tinn red_neuronal_2 = ann[id_red_2];
        float* pesos_p1 = red_neuronal_1.w;
        float* pesos_p2 = red_neuronal_2.w;
        float* pesos_nuevos = (float*) calloc(num_pesos, sizeof(float));
        
        for(int j=0; j < num_pesos; j++) // recorre los pesos de la red
        {
            int azar = (int)rand()%10;
            if(azar<5)
            {
                pesos_nuevos[j] = pesos_p1[j]; 
            }
            else
            {
                pesos_nuevos[j] = pesos_p2[j];
            }
        }
        cont++;
        crossovered_net[i] = xtbuildfrom(27, 243, 9, pesos_nuevos);   
    }
    new_size = cont;
    return crossovered_net;
}

Tinn* mutation(Tinn* red, int size)
{
    srand (time(NULL));
    Tinn* next_gen_net = (Tinn*)calloc(size, sizeof(Tinn));
    int num_pesos = red[0].nw;
    
    for(int i=0; i<size; i++) // avanzar por cada red
    {
        float* pesos = red[i].w;
        float* pesos_nuevos = (float*) calloc(num_pesos, sizeof(float));

        for(int j=0; j<num_pesos; j++) // avanzar por los pesos de la red
        {
            int azar = (int)rand()%10;
            if(azar > 7)
            {
                pesos_nuevos[j] = frand() - 0.5f;
            }
            else{
                pesos_nuevos[j] = pesos[j];
            }
        }
        next_gen_net[i] = xtbuildfrom(27, 243, 9, pesos_nuevos);   
    }

    return next_gen_net;
}

Tinn* duplicate(Tinn* red, int size)
{
    Tinn* duplicated_net = (Tinn*)calloc(size, sizeof(Tinn));
    int cont = 0;
    for(int i=0; i<size/2; i++)
    {
        duplicated_net[cont++] = red[i];
        duplicated_net[cont++] = red[i];
    }
    return duplicated_net;
}

Tinn xtbuildfrom(const int nips, const int nhid, const int nops, float* pesos)
{
    Tinn t;
    // Tinn only supports one hidden layer so there are two biases.
    t.nb = 2;
    t.nw = nhid * (nips + nops);
    t.w = (float*) calloc(t.nw, sizeof(*t.w));
    t.x = t.w + nhid * nips;
    t.b = (float*) calloc(t.nb, sizeof(*t.b));
    t.h = (float*) calloc(nhid, sizeof(*t.h));
    t.o = (float*) calloc(nops, sizeof(*t.o));
    t.nips = nips;
    t.nhid = nhid;
    t.nops = nops;
    from(t, pesos);
    return t;
}

static void from(const Tinn t, float* pesos)
{
    for(int i = 0; i < t.nw; i++) t.w[i] = pesos[i];
    for(int i = 0; i < t.nb; i++) t.b[i] = frand() - 0.5f;
}


static float frand()
{
    return rand() / (float) RAND_MAX;
}

void Wait(cl_command_queue queue)
{
    cl_event wait;
    cl_int status;
    status = clEnqueueMarker(queue, &wait);
    if(status != CL_SUCCESS) fprintf(stderr, "Wait: clEnqueueMarker failed\n");
    status = clWaitForEvents(1, &wait);
    if(status != CL_SUCCESS) fprintf(stderr, "Wait: clWaitForEvents failed\n");
}