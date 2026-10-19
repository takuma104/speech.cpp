#include "sampling.h"

#include <cmath>

#include "json.h"

double uniform(std::mt19937_64 & rng) {
    return (double) rng() * 0x1.0p-64;
}

std::vector<float> gaussian(std::mt19937_64 & rng, size_t n) {
    std::vector<float> out(n);
    constexpr double kTwoPi = 6.283185307179586;
    for (size_t i = 0; i < n; i += 2) {
        // Uniform on (0, 1]: 53 random bits, offset so that the logarithm never sees zero.
        const double u1 = ((rng() >> 11) + 1.0) * 0x1.0p-53, u2 = (rng() >> 11) * 0x1.0p-53;
        const double r = std::sqrt(-2.0 * std::log(u1));
        out[i] = (float) (r * std::cos(kTwoPi * u2));
        if (i + 1 < n) out[i + 1] = (float) (r * std::sin(kTwoPi * u2));
    }
    return out;
}

int pick(const std::vector<double> & weights, double u) {
    double sum = 0;
    for (double w : weights) sum += w;
    const int last = (int) weights.size() - 1;
    double cumulative = 0;
    for (int i = 0; i < last; i++) {
        cumulative += weights[i] / sum;
        if (cumulative > u) return i;
    }
    return last;
}

Error temperature_overflow(float temperature) {
    return Error(Fault::OutOfRange,
                 "the temperature " + json_number(temperature) + " pushes the logits beyond the range of a float; give a temperature nearer 1",
                 "temperature");
}
