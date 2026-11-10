#include <iostream>
#include <math.h>

#define Nz 100
#define Nzp (Nz + 1)

// g++ -std=c++17 -lm -O2 main.cpp -o ekman
// 2026/10/28 Harmen Jonker, version 0.0

float weighted_kick(const float z, const float h_obs, const float sigma)
{
    float zarg = (z-h_obs)/sigma;
    float res = exp(-zarg * zarg / 2.0);
    return res;
}

int main(int argc, char **argv)
{
    if (argc < 8)
    {
        printf("usage: %s expnr duration_in_seconds ug K ukick vkick h_obs sigma\n", argv[0]);
        exit(0);
    }

    float T;

    float u[Nzp];
    float v[Nzp];
    float up[Nzp];
    float vp[Nzp];
    float z[Nzp];

    float Kc = 2.0;
    float ug = 10.0;
    float vg = 0.0;
    float Lz = 500.0;
    float dz = Lz / Nz;
    float htmser;
    int ktmser;
    float fcor = 1e-4;
    float ukick;
    float vkick;
    float sigma;
    float h_obs;

    int expnr = atoi(argv[1]);
    printf("expnr = %d\n", expnr);
    sscanf(argv[2], "%g", &T);
    sscanf(argv[3], "%g", &ug);
    sscanf(argv[4], "%g", &Kc);
    sscanf(argv[5], "%g", &ukick);
    sscanf(argv[6], "%g", &vkick);
    sscanf(argv[7], "%g", &h_obs);
    sscanf(argv[8], "%g", &sigma);


    printf("T=%g\n", T);
    printf("ug=%g\n", ug);
    printf("K=%g\n", Kc);
    printf("ukick=%g\n", ukick);
    printf("vkick=%g\n", vkick);

    htmser = h_obs;
    ktmser = htmser / dz;

    // init half level vars
    for (int k = 0; k < Nzp; k++)
    {
        u[k] = k * ug / Nz;
        v[k] = vg;
        z[k] = k * dz;
    }

    float t_write = 1800.0;
    float t_nextwrite = 0;
    float tmser_write = 60.0;
    float tmser_nextwrite = 0;

    double t;
    up[0] = 0;
    vp[0] = 0;
    up[Nz - 1] = 0;
    vp[Nz - 1] = 0;

    const int maxlen = 1000;
    char name[maxlen];
    snprintf(name, maxlen, "prof.%03d.txt", expnr);
    FILE *fp;
    fp = fopen(name, "w");
    snprintf(name, maxlen, "tmser.%03d.txt", expnr);
    FILE *fp2;
    fp2 = fopen(name, "w");

    float dtmax = 5.0;
    float dt;

    dt = 0.45 * dz * dz / Kc;
    if (dt > dtmax)
        dt = dtmax;

    float Tspinup = 60 * 3600;

    u[Nz] = ug;
    v[Nz] = vg;
    u[0] = 0;
    v[0] = 0;
    up[0] = 0;
    vp[0] = 0;

    // spinup
    t = 0.0;
    while (t < Tspinup)
    {

        for (int k = 1; k < Nz; k++)
        {
            up[k] = Kc * (u[k + 1] - 2 * u[k] + u[k - 1]) / (dz * dz) + fcor * (v[k] - vg);
            vp[k] = Kc * (v[k + 1] - 2 * v[k] + v[k - 1]) / (dz * dz) - fcor * (u[k] - ug);
        }

        for (int k = 1; k < Nz; k++)
        {
            u[k] += dt * up[k];
            v[k] += dt * vp[k];
        }

        t += dt;
    }

    // system is in balance.

    // now kick the system ...

    for (int k = 1; k < Nz; k++)
    {
        u[k] += ukick * weighted_kick(z[k], h_obs, sigma);
        v[k] += vkick * weighted_kick(z[k], h_obs, sigma);
    }

    t = 0.0;

    while (t < T)
    {
        for (int k = 1; k < Nz; k++)
        {
            up[k] = Kc * (u[k + 1] - 2 * u[k] + u[k - 1]) / (dz * dz) + fcor * (v[k] - vg);
            vp[k] = Kc * (v[k + 1] - 2 * v[k] + v[k - 1]) / (dz * dz) - fcor * (u[k] - ug);
        }

        for (int k = 1; k < Nz; k++)
        {
            u[k] += dt * up[k];
            v[k] += dt * vp[k];
        }

        if (t > t_nextwrite)
        {
            fprintf(fp, "#%g\n", t);
            for (int k = 0; k < Nz; k++)
            {
                fprintf(fp, "%+13.5e %+13.5e %+13.5e %+13.5e\n", k * dz, (k + 0.5) * dz, u[k], v[k]);
            }
            fprintf(fp, "\n\n");
            t_nextwrite += t_write;
        }

        if (t > tmser_nextwrite)
        {
            fprintf(fp2, "%g %g %g\n", t, u[ktmser], v[ktmser]);
            tmser_nextwrite += tmser_write;
        }

        t += dt;
    }
    fclose(fp);
    fclose(fp2);
}
