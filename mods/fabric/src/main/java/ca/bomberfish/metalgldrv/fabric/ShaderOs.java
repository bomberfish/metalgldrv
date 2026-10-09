package ca.bomberfish.metalgldrv.fabric;

/**
 * The OS define shader packs see. Iris defines MC_OS_MAC on macOS, and many packs use it to work
 * around Apple's OpenGL 4.1 (turning features off). metalgldrv isn't Apple's OpenGL, so while it is
 * active Iris's getOsString() reports another OS instead (see mixin.IrisStandardMacrosMixin). Only
 * the shader-pack define changes: Minecraft and other mods still detect macOS (input, windowing).
 */
public final class ShaderOs {
    private static volatile String define;  // null: leave Iris alone
    private static volatile boolean logged;

    private ShaderOs() {}

    /** The define to report instead of MC_OS_MAC, or null. Called by the mixin. */
    public static String define() {
        String d = define;
        if (d != null && !logged) {
            logged = true;
            Log.info("shader packs see " + d + " instead of MC_OS_MAC (spoof_shader_os in config/" + Bootstrap.CONFIG_FILE + ")");
        }
        return d;
    }

    /** Sets the define from spoof_shader_os = windows (default) | linux | unknown | off. */
    static void configure(String setting) {
        String v = setting == null ? "windows" : setting.trim().toLowerCase(java.util.Locale.ROOT);
        switch (v) {
            case "", "windows", "true", "on", "yes", "1" -> define = "MC_OS_WINDOWS";
            case "linux" -> define = "MC_OS_LINUX";
            case "unknown" -> define = "MC_OS_UNKNOWN";
            case "off", "false", "no", "0", "mac", "macos" -> define = null;
            default -> {
                Log.warn("spoof_shader_os=" + setting + " isn't windows, linux, unknown or off; using windows");
                define = "MC_OS_WINDOWS";
            }
        }
    }
}
