#include <stdio.h>
#include <stdlib.h>

typedef struct
{
    unsigned char r, g, b;
} Pixel;

typedef struct
{
    int width, height;
    Pixel *data;
} Image;

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
    result->data = (Pixel *)malloc(result->width * result->height * sizeof(Pixel));
    if (!result->data)
    {
        perror("Unable to allocate memory");
        free(result);
        return NULL;
    }

    for (int i = 0; i < result->width * result->height; i++)
    {
        result->data[i].r = (img1->data[i].r + img2->data[i].r) / 2;
        result->data[i].g = (img1->data[i].g + img2->data[i].g) / 2;
        result->data[i].b = (img1->data[i].b + img2->data[i].b) / 2;
    }

    return result;
}

void freeImage(Image *img)
{
    if (img)
    {
        free(img->data);
        free(img);
    }
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s <image1.ppm> <image2.ppm> <output.ppm>\n", argv[0]);
        return EXIT_FAILURE;
    }

    Image *img1 = readPPM(argv[1]);
    Image *img2 = readPPM(argv[2]);

    if (!img1 || !img2)
    {
        freeImage(img1);
        freeImage(img2);
        return EXIT_FAILURE;
    }

    Image *combinedImage = combineImages(img1, img2);
    if (!combinedImage)
    {
        freeImage(img1);
        freeImage(img2);
        return EXIT_FAILURE;
    }

    writePPM(argv[3], combinedImage);

    freeImage(img1);
    freeImage(img2);
    freeImage(combinedImage);

    return EXIT_SUCCESS;
}
