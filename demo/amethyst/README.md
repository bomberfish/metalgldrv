# Amethyst + metalgldrv demo

Builds [Amethyst-iOS](https://github.com/AngelAuraMC/Amethyst-iOS) (pinned at
`f98c413`) with `libmetalgldrv.dylib` embedded and selectable as
**Settings → Renderer → "MetalGL (experimental)"** (also in the per-profile
renderer picker). No Amethyst source lives in this repo, only `patches/`.

## What the patch changes

- `utils.h`: `RENDERER_NAME_METALGL "libmetalgldrv.dylib"`.
- `LauncherPreferences.m` + `en.lproj/Localizable.strings`: new renderer
  choice, shown as "MetalGL (experimental)". Other languages fall back to English.
- `egl_bridge.m`: MetalGL uses the EGL bridge (like ANGLE). `pojavSetWindowHint`
  now records the `GLFW_CONTEXT_VERSION_MAJOR/MINOR`, `GLFW_OPENGL_PROFILE` and
  `GLFW_OPENGL_FORWARD_COMPAT` hints.
- `ctxbridges/gl_bridge.m`: for MetalGL, EGL is loaded with dlsym from
  `@rpath/libmetalgldrv.dylib` (not `libtinygl4angle.dylib`). The config uses
  `EGL_OPENGL_BIT`, the bridge calls `eglBindAPI(EGL_OPENGL_API)`, and the
  context is created with `EGL_CONTEXT_MAJOR/MINOR_VERSION`, the profile mask
  and forward-compat flag the game asked GLFW for. With no hint it uses
  `EGL_CONTEXT_CLIENT_VERSION 3`, which gives a 4.6 compatibility context.

LWJGL loads GL from `-Dorg.lwjgl.opengl.libname=libmetalgldrv.dylib`, which
resolves to `<app>/Frameworks`. On iOS, LWJGL takes the macOS path, so it
resolves every `gl*` entry point with **`dlsym` on that library**. It never
calls `eglGetProcAddress`. It does call `OSMesaGetProcAddress` first, but only
if the library exports it.

## Build

```sh
PROFILE=~/Library/Developer/Xcode/UserData/Provisioning\ Profiles/<uuid>.mobileprovision \
DEVICE=<udid> demo/amethyst/build.sh            # fetch metalgl amethyst sign install
demo/amethyst/build.sh metalgl amethyst sign    # rebuild after driver changes (no re-fetch)
DEVICE=<udid> demo/amethyst/build.sh launch     # launch with console attached
```

Steps: `fetch` clones Amethyst, resets it to the pinned commit, applies
`patches/*.patch` with `git am`, then initializes the submodules. `metalgl`
builds the dylib with Ninja. `amethyst` runs `gmake payload` and then copies
the dylib into `Frameworks/`. `sign` re-signs every nested Mach-O and the app
with the profile's entitlements. `install` and `launch` use `xcrun devicectl`.

Environment variables (defaults in brackets):

- `AMETHYST_WORK` [`~/metalgldrv-deps/amethyst-work`]: the work tree. The
  script owns it and resets it on `fetch`.
- `METALGL_BUILD_DIR` [`$AMETHYST_WORK-metalgldrv-build`] and
  `METALGL_BUILD_TYPE` [`RelWithDebInfo`]: the driver build lives outside the
  repo, so it never touches `build-ios/`.
- `AMETHYST_REPO` and `AMETHYST_COMMIT`: the source to clone and the pinned commit.
- `BOOTJDK` [zulu-8 `bin/`]: JDK 8, needed for the Java part.
- `PROFILE`: the provisioning profile. Without it, the app keeps the
  Makefile's ldid ad-hoc signature.
- `SIGN_IDENTITY` [found from the profile's certificate].
- `BUNDLE_ID` [taken from the profile].
- `DEVICE`: the device UDID or name, for `install` and `launch`.

Prerequisites: Xcode with the iPhoneOS SDK, plus the Homebrew packages `cmake`,
`ninja`, `gmake` and `ldid`, plus JDK 8. A first build downloads about 90 MB of
iOS JREs and about 800 MB of submodules. The work tree ends up around 2.6 GB.

Notes:

- The `DBNumberedSlider` and `fishhook` submodule repos (owned by
  `khanhduytran0`) no longer exist. The script fetches the same pinned commits
  from their fork networks (`immago/DBNumberedSlider` and `facebook/fishhook`).
- Signing uses exactly the entitlements the profile grants. Amethyst's
  `com.apple.private.*` entitlements are dropped, because a development
  certificate can't sign them.
- Running Java still needs JIT, the same as stock Amethyst: attach a debugger,
  for example with StikDebug.
- Logs are written to `Documents/latestlog.txt` in the app container. The
  bridge logs `EGLBridge: MetalGL context requested: …`. To fetch the log:
  `xcrun devicectl device copy from --device <udid> --domain-type appDataContainer --domain-identifier <bundle id> --source Documents/latestlog.txt --destination .`

## Regenerating the patch

Commit your changes on the `metalgl` branch in the work tree, then run
`git format-patch -o demo/amethyst/patches f98c4133c72d39e06ceac42b57cc4477c52105fc`
(delete the old patch files first).
