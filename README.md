# metalgldrv

An OpenGL 4.6 driver (core and compatibility profiles) for Apple platforms, written from scratch on
Metal 4. It has its own GLSL compiler, which translates to Metal Shading Language, and uses no Mesa or
other third-party code.

It runs:

* **Minecraft: Java Edition on iOS/iPadOS** through the [Amethyst](https://github.com/AngelAuraMC/Amethyst-iOS)
  launcher (EGL), including Sodium and Iris shader packs.
* **Minecraft: Java Edition on macOS** through a Fabric mod, with the same mods and shader packs.
* **Windows OpenGL games in CrossOver** on macOS, which otherwise stop at Apple's GL 4.1. DOOM (2016)'s
  OpenGL renderer is the reference game; it plays at about 45 fps on ultra at 1600x900 on an M4 Pro.

## Conformance

The driver passes the public Khronos conformance suites (VK-GL-CTS 4.6.8.1) with no failures:

| Case list | Pass | Not supported | Fail |
|---|---|---|---|
| KHR-GL46 | 18061 | 1652 | 0 |
| KHR-Single-GL46 | 11340 | 8 | 0 |
| KHR-NoContext | 8 | 0 | 0 |

"Not supported" cases cover optional features. Results for every configuration, surface size and
accuracy profile, on Mac and iPad, are in [tests/cts/reports/LATEST.md](tests/cts/reports/LATEST.md);
[NOT_SUPPORTED.md](tests/cts/reports/NOT_SUPPORTED.md) explains the not-supported cases. The GTF suites
aren't public and aren't covered.

## Requirements

* A GPU with Metal 4: Apple silicon Macs (M1 and later) and iPads/iPhones with a Metal 4 GPU. The driver
  steps aside on anything older.
* macOS 26 / iOS 26 or later.
* To build: Xcode with the macOS 26 / iOS 26 SDKs, CMake and Ninja.

## Building

```sh
# macOS (arm64), with the host tests
cmake -G Ninja -B build && ninja -C build

# macOS universal (arm64 + x86_64): needed for CrossOver, whose Wine runs under Rosetta,
# and used by the Fabric mod
cmake -G Ninja -B build-universal -DMGL_UNIVERSAL=ON -DCMAKE_BUILD_TYPE=Release && ninja -C build-universal

# iOS / iPadOS
cmake -G Ninja -B build-ios -DCMAKE_TOOLCHAIN_FILE=cmake/ios.cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo && ninja -C build-ios
```

The output is `libmetalgldrv.dylib`, which exports EGL 1.5, every GL entry point and (on macOS) CGL.

## Using it

### Minecraft on macOS (Fabric)

Build the mod with `mods/fabric/build.sh` and put `dist/metalgldrv-fabric-<version>.jar` in the
instance's `mods/` folder. No JVM flags or launcher settings are needed. It covers Minecraft 1.20.2
through 26.3. With Mod Menu and Cloth Config installed (Fabulously Optimized has both), Mods ->
metalgldrv -> Configure sets the accuracy/speed profile, render resolution, MetalFX upscaling and the
OS shader packs see. Otherwise edit the instance's `config/metalgldrv.properties`. See
[mods/fabric/README.md](mods/fabric/README.md) and [docs/minecraft-macos.md](docs/minecraft-macos.md),
which also describes manual setups without the mod.

### Your own app

Link `libmetalgldrv.dylib` in place of OpenGL.framework (CGL, NSOpenGL, GLFW and SDL code keeps
working unchanged), switch to it at run time with `metalgldrv_activate()`, or use EGL (macOS and iOS).
`cmake --install` provides headers, a CMake package (`find_package(metalgldrv)`) and pkg-config. See
[docs/porting.md](docs/porting.md).

### Minecraft on iOS (Amethyst)

`demo/amethyst` builds Amethyst with the driver embedded and selectable as a renderer. The repo contains
only the patches, not Amethyst's source. See [demo/amethyst/README.md](demo/amethyst/README.md).

### Windows games in CrossOver

```sh
ditto "/Applications/CrossOver.app" ~/Applications/CrossOver-MetalGL.app   # patch a copy
scripts/crossover-install.sh --app ~/Applications/CrossOver-MetalGL.app
scripts/crossover-install.sh --app ~/Applications/CrossOver-MetalGL.app --uninstall   # undo
```

This points that copy's `winemac.so` at the universal build, so its bottles use metalgldrv for
OpenGL. It works the same with [NotProton](https://github.com/NotProtonNot/NotProton)'s CrossOver
runner (`--wine-root`). See [docs/crossover.md](docs/crossover.md), and [docs/doom2016.md](docs/doom2016.md)
for DOOM specifics.

Games launched by Steam/NotProton/CrossOver don't see your shell environment. Put driver settings in
`~/Library/Application Support/metalgldrv/metalgldrv.conf` instead, one `METALGLDRV_NAME=value` per
line. The file applies to every process that loads the driver, unless that process's environment
already sets the variable.

## Accuracy and speed profiles

The default mode is conformant. Other profiles trade in either direction:

| Setting | Profile | Effect |
|---|---|---|
| `METALGLDRV_STRICT=2` | super strict | Strict, plus the most expensive emulations (e.g. post-depth coverage), and no fused multiply-add across statements. |
| `METALGLDRV_STRICT=1` | strict | Accuracy features that conformance doesn't need but that cost performance: bounds-checked texel fetches everywhere, precise math functions, IEEE-safe shader arithmetic (no reassociation), the full tessellation emulation, and more. |
| (unset) | default | Conformant, fast paths that give identical results. Shader math may be reassociated and fused as GLSL allows (NaN and infinity still behave). |
| `METALGLDRV_TURBOHAX=1` | turbo | Not conformant: fast math (no NaN/infinity guarantees), anisotropy capped at 4x. |
| `METALGLDRV_SPEEDDEMON=1` | speed demon | Turbo, plus: query results never wait for the GPU, `glFinish` doesn't wait, no robust-access bounds checks, anisotropy capped at 2x. |

Strict wins when several are set.

## Lower render resolution and MetalFX upscaling

* `METALGLDRV_USE_METALFX=1`: the window is presented at the display's native resolution, and when
  the game renders smaller, MetalFX's spatial scaler upscales the image (instead of plain scaling).
* `METALGLDRV_RENDER_SCALE=<0.25..1>` (Minecraft and other macOS GL apps, not CrossOver): the game
  renders at that fraction of the window's pixel size. `0.5` is half the width and height, a quarter
  of the pixels. The driver reports the smaller size to GLFW / SDL, so the game sizes all of its
  render targets to it. Set both on the mod's config screen, or in `config/metalgldrv.properties`:

  ```properties
  METALGLDRV_RENDER_SCALE=0.5
  METALGLDRV_USE_METALFX=1
  ```

* CrossOver: Wine sizes windows itself, so the render scale doesn't apply. Windows there are already
  half resolution when the bottle's Retina mode is off (the default in NotProton): DOOM's 1600x900
  window covers 3200x1800 display pixels. `METALGLDRV_USE_METALFX=1` upscales those with MetalFX. A
  game's own resolution-scale setting lowers it further.
* Only the spatial scaler is supported. MetalFX's temporal upscaling needs each frame's motion vectors,
  depth and camera jitter, which OpenGL games don't hand to the driver.

## Configuration and debugging

Every `METALGLDRV_*` variable is listed in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) under
"Environment variables". The common ones:

* `METALGLDRV_LOG_FILE=<path>`: driver log. `METALGLDRV_LOG_UNSUPPORTED=1` warns once about
  unsupported calls and GL errors, and dumps failing shaders to `METALGLDRV_DUMP_DIR`.
* `METALGLDRV_STATS=1`: memory, GPU busy time and event counters every 5 seconds.
  `METALGLDRV_PROFILE_PASSES=1` adds GPU time per render pass.
* `METALGLDRV_HITCH_MS=<n>`: for frames slower than n ms, logs where the time went (GL calls per
  thread, shader compiles, residency).
* `METALGLDRV_APP_PROFILE`, `METALGLDRV_VENDOR`, ...: per-application profiles and overrides
  (`src/gl/app_profiles.cpp`).

## Tests

```sh
./build/tests/mgltest                       # GL-level host tests
./build/tests/linktest                      # an app linked in place of OpenGL.framework
tests/cts/run.sh                            # Khronos CTS (tests/cts/setup.sh fetches and builds it)
tests/cts/run.sh -l gl46-khr-single         # other case lists; --ios runs on a connected device
tests/fabric-mod/run.sh                     # Fabric mod loader scenarios
```

## Documentation

| Document | Contents |
|---|---|
| [docs/porting.md](docs/porting.md) | Using the driver in your own application (macOS and iOS) |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Layout, design, every subsystem, environment variables, application profiles |
| [docs/glsl-compiler.md](docs/glsl-compiler.md) | The GLSL → MSL compiler |
| [docs/minecraft-macos.md](docs/minecraft-macos.md) | Minecraft on macOS |
| [docs/crossover.md](docs/crossover.md) | CrossOver / Wine |
| [docs/doom2016.md](docs/doom2016.md) | DOOM (2016): what it needs and what was done for it |

## License

Not yet chosen.
