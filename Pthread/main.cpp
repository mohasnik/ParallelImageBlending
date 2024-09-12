#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <omp.h>

#include "x86intrin.h"
#include <sys/time.h>

#include "../lib/primitives.h"

#define IMAGE1_ADDRESS "../img/q4_img01.png"
#define IMAGE2_ADDRESS "../img/q4_img02.png"

#define ALPHA_INVERSED 4
#define THREADS_NUM 20

typedef struct timeval timeVal;

typedef struct TheadArguments
{
    int threadNum;

    int alphaInversed;
    cv::Mat* img1;
    cv::Mat* img2;
    
    unsigned char* resultImg;

} ThreadArguments;

void* AddImages(void* arg)
{

    ThreadArguments* tArgs = (ThreadArguments*)arg;
    
    
    // int startIndex, stopIndex;    
        
    int row, col, idxImg2, idx;
    int img1NumOfCols = tArgs->img1->cols;
    int img2NumOfCols = tArgs->img2->cols;
    int img1NumOfRows = tArgs->img1->rows;
    int img2NumOfRows = tArgs->img2->rows;
    unsigned char res;

    int totalSize = img1NumOfRows * img1NumOfCols;
    int startIndex = totalSize/THREADS_NUM * (tArgs->threadNum);
    
    if( tArgs->threadNum == 0)
        startIndex = 0;
    
    int stopIndex = totalSize/THREADS_NUM * (tArgs->threadNum + 1);
    
    if(stopIndex > totalSize)
        stopIndex = totalSize;

    // printf("T %d : (%d, %d)\n", tArgs->threadNum, startIndex, stopIndex);

    for (int i = startIndex; i < stopIndex; i++)
    {
        idx = i;

        row = (idx)/(img1NumOfCols);
        col = (idx) % (img1NumOfCols);

        // printf("index : %d -> row : %d, col : %d\n", idx, row, col);

        res = tArgs->img1->data[idx];

        if(row <= img2NumOfRows && col <= img2NumOfCols)
        {
            idxImg2 = row * img2NumOfCols + col;
            res += ((tArgs->img2->data[idxImg2]) / tArgs->alphaInversed);
            
            if(res > 255 || res <= 0)
                res = 255;

        }
            
        tArgs->resultImg[idx] = res;

    }

    pthread_exit(NULL);
    
    
}


cv::Mat imgAdditionSerial(cv::Mat img1, cv::Mat img2, long* execTime, int alphaInversed)
{
    cv::Mat resultImg(img1.rows, img1.cols, CV_8U);
    unsigned char* resultImgData = (unsigned char*) resultImg.data;
    unsigned char* img1Data = (unsigned char*) img1.data;
    unsigned char* img2Data = (unsigned char*) img2.data;

    timeVal start, end;

    // ----------------------- calculation started ---------------------------/
    gettimeofday(&start, NULL);
    for (int row = 0; row < img1.rows; row++)
    {
        for (int col = 0; col < img1.cols; col++)
        {
            int idx1 = row * img1.cols + col;
            int res = img1Data[idx1];

            if(row <= img2.rows && col <= img2.cols)
            {
                int idx2 = row * img2.cols + col;
                res += (img2Data[idx2] / alphaInversed);
                
                if(res > 255)
                    res = 255;
                
            }

            resultImgData[idx1] = res;
            
        }
        
    }
    gettimeofday(&end, NULL);
    // ----------------------- calculation finished ---------------------------/
    *execTime = end.tv_usec - start.tv_usec;

    return resultImg;
    
}

cv::Mat imgAdditionParallel(cv::Mat img1, cv::Mat img2, long* execTime, int alphaInversed)
{
    cv::Mat resultImage(img1.rows, img1.cols, CV_8U);

    unsigned char* resImgData = (unsigned char*) resultImage.data;
    unsigned char* img1Data = (unsigned char*) img1.data;
    unsigned char* img2Data = (unsigned char*) img2.data;
    timeVal start, end;
    
    // int totalSize = img1.cols * img1.rows;

    // ----------------------- calculation started ---------------------------/
    gettimeofday(&start, NULL);
    pthread_t threads[THREADS_NUM];
    ThreadArguments tArgs[THREADS_NUM];
    

    for(int i = 0; i < THREADS_NUM; i++)
    {
        tArgs[i] = {i, alphaInversed, &img1, &img2, resImgData};
        pthread_create(&threads[i], NULL, AddImages, (void*)(&(tArgs[i])));
    }
    

    for(int i = 0; i < THREADS_NUM; i++)
        pthread_join(threads[i], NULL);
    

    gettimeofday(&end, NULL);
    // ----------------------- calculation finished ---------------------------/
    *execTime = end.tv_usec - start.tv_usec;

    return resultImage;


}





int main()
{
    const int numOfExec = 1000;

    long serialExecTime, parallelExecTime;
    cv::Mat img1 = cv::imread(IMAGE1_ADDRESS, cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(IMAGE2_ADDRESS, cv::IMREAD_GRAYSCALE);
    float  speedup, speedupAvg = 0;

    printf("size : %d * %d -> totalIndex : %d\n", img1.rows, img1.cols, img1.rows * img1.cols);
    for(int i = 0; i < numOfExec; i++)
    {
        cv::Mat resultImgSerial = imgAdditionSerial(img1, img2, &serialExecTime, ALPHA_INVERSED);
        cv::Mat resultImgParallel = imgAdditionParallel(img1, img2, &parallelExecTime, ALPHA_INVERSED);
        speedup = SPEED_UP(serialExecTime, parallelExecTime);

        printf("execution No.%i\n", i);
        printf("Serial exec. Time : %li us\n", serialExecTime);
        printf("Parallel exec. Time : %li us\n", parallelExecTime);
        printf("-> speedup : %0.4f\n\n", speedup);


        if(speedup < 0)
            continue;

        speedupAvg += speedup;

        if(i == numOfExec-1)
        {
            cv::namedWindow("serial", cv::WINDOW_AUTOSIZE);
            cv::imshow("serial", resultImgSerial);

            cv::namedWindow("parallel", cv::WINDOW_AUTOSIZE);
            cv::imshow("parallel", resultImgParallel);
            cv::waitKey(0);
        }
        
    }

    speedupAvg /= numOfExec;

    printf("***********\n");
    printf("Average Speedup in %i executions : %0.4f\n", numOfExec, speedupAvg);
    printf("***********\n");



    return 0;

}