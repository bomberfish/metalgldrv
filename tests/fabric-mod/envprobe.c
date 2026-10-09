// Harness.getenv0: the process's C environment, which the driver's JNI_OnLoad writes
// to (Java's System.getenv is a snapshot taken at JVM start).
#include <jni.h>
#include <stdlib.h>

JNIEXPORT jstring JNICALL Java_Harness_getenv0(JNIEnv* env, jclass cls, jstring name) {
    const char* n = (*env)->GetStringUTFChars(env, name, NULL);
    const char* v = n ? getenv(n) : NULL;
    if (n) (*env)->ReleaseStringUTFChars(env, name, n);
    return v ? (*env)->NewStringUTF(env, v) : NULL;
}
