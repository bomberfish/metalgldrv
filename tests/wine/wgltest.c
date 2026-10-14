// Minimal Win32 + WGL test without SDK headers: 4.6 core context, GL_VERSION, clear, read back.
typedef void* HANDLE; typedef HANDLE HWND, HDC, HGLRC, HINSTANCE, HMODULE; typedef unsigned int UINT; typedef unsigned long DWORD;
typedef long long LPARAM, LRESULT; typedef unsigned long long WPARAM; typedef int BOOL;
#define API __attribute__((ms_abi)) __declspec(dllimport)
typedef LRESULT (__attribute__((ms_abi)) *WNDPROC)(HWND, UINT, WPARAM, LPARAM);
typedef struct { UINT style; WNDPROC proc; int a, b; HINSTANCE inst; HANDLE icon, cursor, bg; const char* menu; const char* cls; } WNDCLASSA;
typedef struct { unsigned short nSize, nVersion; DWORD dwFlags; unsigned char iPixelType, cColorBits, r, rs, g, gs, b, bs, a, as, acc, accr, accg, accb, acca, cDepthBits, cStencilBits, aux, layer, res; DWORD lm, vm, dm; } PFD;
typedef struct { HWND h; UINT m; WPARAM w; LPARAM l; DWORD t; long x, y; DWORD p; } MSG;
API HMODULE GetModuleHandleA(const char*); API unsigned short RegisterClassA(const WNDCLASSA*);
API HWND CreateWindowExA(DWORD, const char*, const char*, DWORD, int, int, int, int, HWND, HANDLE, HINSTANCE, void*);
API HDC GetDC(HWND); API LRESULT DefWindowProcA(HWND, UINT, WPARAM, LPARAM); API BOOL PeekMessageA(MSG*, HWND, UINT, UINT, UINT);
API LRESULT DispatchMessageA(const MSG*); API int wsprintfA(char*, const char*, ...);
API int ChoosePixelFormat(HDC, const PFD*); API BOOL SetPixelFormat(HDC, int, const PFD*); API BOOL SwapBuffers(HDC);
API HGLRC wglCreateContext(HDC); API BOOL wglMakeCurrent(HDC, HGLRC); API void* wglGetProcAddress(const char*);
API const unsigned char* glGetString(UINT); API void glClearColor(float, float, float, float); API void glClear(UINT);
API void glViewport(int, int, int, int); API void glReadPixels(int, int, int, int, UINT, UINT, void*); API UINT glGetError(void);
API HANDLE GetStdHandle(DWORD); API BOOL WriteFile(HANDLE, const void*, DWORD, DWORD*, void*); API void Sleep(DWORD); API void ExitProcess(UINT);
typedef HGLRC (__attribute__((ms_abi)) *CCA)(HDC, HGLRC, const int*);
static LRESULT __attribute__((ms_abi)) proc(HWND h, UINT m, WPARAM w, LPARAM l) { return DefWindowProcA(h, m, w, l); }
static void out(const char* s) { DWORD n = 0, len = 0; while (s[len]) len++; WriteFile(GetStdHandle((DWORD)-11), s, len, &n, 0); }
void __attribute__((ms_abi)) start(void) {
    char buf[512];
    WNDCLASSA wc = {0x20, proc, 0, 0, GetModuleHandleA(0), 0, 0, 0, 0, "wgltest"};
    RegisterClassA(&wc);
    HWND w = CreateWindowExA(0, "wgltest", "wgltest", 0x10CF0000, 100, 100, 320, 240, 0, 0, wc.inst, 0);
    HDC dc = GetDC(w);
    PFD pfd = {sizeof(PFD), 1, 0x25, 0, 32}; pfd.cDepthBits = 24;
    if (!SetPixelFormat(dc, ChoosePixelFormat(dc, &pfd), &pfd)) { out("SetPixelFormat failed\n"); ExitProcess(2); }
    wglMakeCurrent(dc, wglCreateContext(dc));
    wsprintfA(buf, "legacy GL_VERSION: %s\n", glGetString(0x1F02)); out(buf);
    CCA cca = (CCA)wglGetProcAddress("wglCreateContextAttribsARB");
    int at[] = {0x2091, 4, 0x2092, 6, 0x9126, 1, 0};
    HGLRC core = cca ? cca(dc, 0, at) : 0;
    if (!core) { out("4.6 core context: not available\n"); ExitProcess(3); }
    wglMakeCurrent(dc, core);
    wsprintfA(buf, "core GL_VERSION: %s\nGL_RENDERER: %s\n", glGetString(0x1F02), glGetString(0x1F01)); out(buf);
    unsigned char px[4] = {0};
    glViewport(0, 0, 320, 240);
    for (int i = 0; i < 30; i++) { MSG m; while (PeekMessageA(&m, 0, 0, 0, 1)) DispatchMessageA(&m);
        glClearColor(1, 0, 0, 1); glClear(0x4000); glReadPixels(160, 120, 1, 1, 0x1908, 0x1401, px); SwapBuffers(dc); Sleep(16); }
    wsprintfA(buf, "pixel: %d %d %d %d, glGetError %x\n", px[0], px[1], px[2], px[3], glGetError()); out(buf);
    ExitProcess(px[0] == 255 && px[1] == 0 ? 0 : 4);
}
int _fltused = 0;
