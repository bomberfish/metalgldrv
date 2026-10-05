#version 330 core
#define SHADOW_QUALITY 2
#define MUL(a, b) ((a) * (b))
#define SQR(x) MUL(x, x)
#define CONCAT_TEST 1
#define LONG_MACRO(v) \
    (v + 1.0 \
     + 2.0)
#if SHADOW_QUALITY > 1 && defined(CONCAT_TEST)
  #define SAMPLES 16
#elif SHADOW_QUALITY == 1
  #define SAMPLES 4
#else
  #define SAMPLES 1
#endif
#ifndef UNDEFINED_THING
  #define HAS_THING 0
#endif
#if !defined UNDEFINED_THING && (SAMPLES * 2 >= 32) || 0
const float k = 1.0;
#else
const float k = 2.0;
#endif
#undef MUL
#define MUL(a, b) ((a) + (b))
out vec4 c;
void main() {
    float x = SQR(2.0) + LONG_MACRO(1.0) + float(SAMPLES) + k + float(__LINE__) + float(__VERSION__);
#if HAS_THING
    x = 0.0;
#endif
    c = vec4(x);
}
