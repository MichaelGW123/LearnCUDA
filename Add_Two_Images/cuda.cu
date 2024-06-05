#include <stdio.h>
#include <stdlib.h>

// Structure to hold RGB values of a pixel
typedef struct
{
    unsigned char r, g, b;
} Pixel;

// Structure to hold an image
typedef struct
{
    int width, height;
    Pixel *data;
} Image;

// Function to read a PPM image file
Image *readPPM(const char *filename)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        perror("Unable to open file");
        return NULL;
    }

    char format[3];
    fscanf(fp, "%s", format);

    if (format[0] != 'P' || format[1] != '6')
    {
        fprintf(stderr, "Invalid image format (must be 'P6')\n");
        fclose(fp);
        return NULL;
    }

    Image *img = (Image *)malloc(sizeof(Image));
    if (!img)
    {
        perror("Unable to allocate memory");
        fclose(fp);
        return NULL;
    }

    fscanf(fp, "%d %d", &img->width, &img->height);
    int max_val;
    fscanf(fp, "%d", &max_val);

    while (fgetc(fp) != '\n')
        ; // Skip the remaining header data

    img->data = (Pixel *)malloc(img->width * img->height * sizeof(Pixel));
    if (!img->data)
    {
        perror("Unable to allocate memory");
        free(img);
        fclose(fp);
        return NULL;
    }

    fread(img->data, sizeof(Pixel), img->width * img->height, fp);
    fclose(fp);
    return img;
}

// Function to write a PPM image file
void writePPM(const char *filename, Image *img)
{
    FILE *fp = fopen(filename, "wb");
    if (!fp)
    {
        perror("Unable to open file");
        return;
    }

    fprintf(fp, "P6\n%d %d\n255\n", img->width, img->height);
    fwrite(img->data, sizeof(Pixel), img->width * img->height, fp);
    fclose(fp);
}

// CUDA kernel to combine two images pixel-by-pixel
__global__ void combineImagesKernel(Pixel *img1, Pixel *img2, Pixel *result, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        result[i].r = (img1[i].r + img2[i].r) / 2;
        result[i].g = (img1[i].g + img2[i].g) / 2;
        result[i].b = (img1[i].b + img2[i].b) / 2;
    }
}

// Function to combine two images using CUDA
Image *combineImages(Image *img1, Image *img2)
{
    if (img1->width != img2->width || img1->height != img2->height)
    {
        fprintf(stderr, "Images must be of the same dimensions\n");
        return NULL;
    }

    Image *result = (Image *)malloc(sizeof(Image));
    if (!result)
    {
        perror("Unable to allocate memory");
        return NULL;
    }

    result->width = img1->width;
    result->height = img1->height;
    int size = result->width * result->height;
    result->data = (Pixel *)malloc(size * sizeof(Pixel));
    if (!result->data)
    {
        perror("Unable to allocate memory");
        free(result);
        return NULL;
    }

    // Allocate device memory
    Pixel *d_img1, *d_img2, *d_result;
    cudaMalloc((void **)&d_img1, size * sizeof(Pixel));
    cudaMalloc((void **)&d_img2, size * sizeof(Pixel));
    cudaMalloc((void **)&d_result, size * sizeof(Pixel));

    // Copy data from host to device
    cudaMemcpy(d_img1, img1->data, size * sizeof(Pixel), cudaMemcpyHostToDevice);
    cudaMemcpy(d_img2, img2->data, size * sizeof(Pixel), cudaMemcpyHostToDevice);

    // Calculate block size and number of blocks
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    // Launch kernel to combine images
    combineImagesKernel<<<numBlocks, blockSize>>>(d_img1, d_img2, d_result, size);

    // Copy result from device to host
    cudaMemcpy(result->data, d_result, size * sizeof(Pixel), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_img1);
    cudaFree(d_img2);
    cudaFree(d_result);

    return result;
}

// Function to free allocated image memory
void freeImage(Image *img)
{
    if (img)
    {
        free(img->data);
        free(img);
    }
}

// Main function
int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s <image1.ppm> <image2.ppm> <output.ppm>\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Read the input images
    Image *img1 = readPPM(argv[1]);
    Image *img2 = readPPM(argv[2]);

    if (!img1 || !img2)
    {
        freeImage(img1);
        freeImage(img2);
        return EXIT_FAILURE;
    }

    // Combine the images using CUDA
    Image *combinedImage = combineImages(img1, img2);
    if (!combinedImage)
    {
        freeImage(img1);
        freeImage(img2);
        return EXIT_FAILURE;
    }

    // Write the combined image to the output file
    writePPM(argv[3], combinedImage);

    // Free allocated memory
    freeImage(img1);
    freeImage(img2);
    freeImage(combinedImage);

    return EXIT_SUCCESS;
}
