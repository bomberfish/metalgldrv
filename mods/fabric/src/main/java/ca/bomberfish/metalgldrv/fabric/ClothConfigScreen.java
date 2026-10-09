package ca.bomberfish.metalgldrv.fabric;

import com.terraformersmc.modmenu.api.ConfigScreenFactory;
import java.lang.reflect.Array;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.nio.file.Path;
import java.util.function.Consumer;
import java.util.function.Function;
import net.fabricmc.loader.api.FabricLoader;
import net.fabricmc.loader.api.MappingResolver;

/**
 * The config screen (Mod Menu's "Configure" button), built with Cloth Config when it's installed.
 *
 * The mod has no compile-time dependency on Minecraft, so that one jar serves 1.20.2 to 26.x: Minecraft's
 * runtime class names are intermediary ones up to 1.21 and Mojang's from 26.1. Cloth Config and Minecraft's
 * text class are therefore reached through reflection, and Mod Menu's ConfigScreenFactory (whose create()
 * takes a Screen) is implemented with a Proxy.
 */
final class ClothConfigScreen {
    private ClothConfigScreen() {}

    private static final String[] PROFILES = {"Default", "Strict", "Super strict", "TURBOHAX", "SPEEDDEMON"};
    private static final String[] SCALES = {"100%", "75%", "50%", "25%"};
    private static final String[] SHADER_OS = {"Windows", "Linux", "Unknown", "macOS (no spoofing)"};

    /** Mod Menu's factory: Cloth Config's screen, or no screen when Cloth Config isn't installed. */
    static ConfigScreenFactory<?> factory() {
        ClassLoader cl = ClothConfigScreen.class.getClassLoader();
        return (ConfigScreenFactory<?>) Proxy.newProxyInstance(cl, new Class<?>[] {ConfigScreenFactory.class}, (proxy, m, args) -> {
            switch (m.getName()) {
                case "create":
                    return clothAvailable() ? build(args[0]) : null;
                case "hashCode":
                    return System.identityHashCode(proxy);
                case "equals":
                    return proxy == args[0];
                case "toString":
                    return "metalgldrv config screen factory";
                default:
                    return null;
            }
        });
    }

    static boolean clothAvailable() {
        try {
            Class.forName("me.shedaniel.clothconfig2.api.ConfigBuilder", false, ClothConfigScreen.class.getClassLoader());
            return true;
        } catch (ClassNotFoundException e) {
            return false;
        }
    }

    private static Object build(Object parent) throws Exception {
        Path path = FabricLoader.getInstance().getConfigDir().resolve(Bootstrap.CONFIG_FILE);
        ConfigFile cfg = new ConfigFile(path);
        Class<?> builderClass = Class.forName("me.shedaniel.clothconfig2.api.ConfigBuilder", true, ClothConfigScreen.class.getClassLoader());
        Object builder = builderClass.getMethod("create").invoke(null);
        call(builder, "setParentScreen", parent);
        call(builder, "setTitle", text("metalgldrv"));
        call(builder, "setSavingRunnable", (Runnable) cfg::save);
        Object entries = call(builder, "entryBuilder");

        // ---- General ----
        Object general = call(builder, "getOrCreateCategory", text("General"));
        add(general, toggle(entries, "Use metalgldrv", !Bootstrap.isFalse(orDefault(cfg.get("enabled"), "true")), true,
                            v -> cfg.set("enabled", v ? "true" : "false"), true,
                            "Off: Minecraft uses Apple's OpenGL 4.1."));
        add(general, selector(entries, "Accuracy / speed profile", PROFILES, currentProfile(cfg), "Default", true, v -> setProfile(cfg, v),
                              "Default: conformant and fast.",
                              "Strict / Super strict: extra accuracy that costs performance.",
                              "TURBOHAX / SPEEDDEMON: give up accuracy for speed (fast math,",
                              "lower anisotropic filtering; SPEEDDEMON also never waits on queries)."));
        add(general, selector(entries, "Render resolution", SCALES, currentScale(cfg), "100%", true,
                              v -> cfg.set("METALGLDRV_RENDER_SCALE", v.equals("100%") ? null : scaleValue(v)),
                              "Renders at this fraction of the window's width and height",
                              "and scales the image up to the window. 50% renders a quarter of the pixels."));
        add(general, toggle(entries, "MetalFX upscaling", cfg.isOn("METALGLDRV_USE_METALFX"), false,
                            v -> cfg.set("METALGLDRV_USE_METALFX", v ? "1" : null), true,
                            "Upscale with MetalFX instead of plain scaling when rendering below",
                            "the display's resolution (Render resolution below 100%)."));

        // ---- Shaders ----
        Object shaders = call(builder, "getOrCreateCategory", text("Shader packs"));
        add(shaders, selector(entries, "OS shown to shader packs", SHADER_OS, currentShaderOs(cfg), "Windows", false, v -> {
            String s = shaderOsValue(v);
            cfg.set("spoof_shader_os", s);
            if (System.getProperty("metalgldrv.activation") != null) ShaderOs.configure(s);  // live, while the driver is active
        }, "Many shader packs turn features off on macOS (MC_OS_MAC) to work around",
           "Apple's OpenGL. metalgldrv doesn't need that. Applies when the shader pack reloads."));

        // ---- Debugging ----
        Object debug = call(builder, "getOrCreateCategory", text("Debugging"));
        add(debug, toggle(entries, "Log unsupported features", cfg.isOn("METALGLDRV_LOG_UNSUPPORTED"), false,
                          v -> cfg.set("METALGLDRV_LOG_UNSUPPORTED", v ? "1" : null), true,
                          "Logs GL errors and unsupported calls once each, and saves failing shaders."));
        add(debug, toggle(entries, "Performance stats", cfg.isOn("METALGLDRV_STATS"), false,
                          v -> cfg.set("METALGLDRV_STATS", v ? "1" : null), true,
                          "Logs memory, GPU busy time and driver counters every 5 seconds."));
        add(debug, toggle(entries, "Log slow frames", cfg.get("METALGLDRV_HITCH_MS") != null, false,
                          v -> cfg.set("METALGLDRV_HITCH_MS", v ? "150" : null), true,
                          "For frames slower than 150 ms, logs where the time went."));

        return call(builder, "build");
    }

    // ---- settings <-> file ----

    private static String orDefault(String v, String d) {
        return v == null ? d : v;
    }

    private static String currentProfile(ConfigFile cfg) {
        String strict = ConfigFile.lower(cfg.get("METALGLDRV_STRICT"));
        if (strict != null && !strict.isEmpty() && !Bootstrap.isFalse(strict)) {
            return strict.startsWith("super") || strict.equals("2") ? "Super strict" : "Strict";
        }
        if (cfg.isOn("METALGLDRV_SPEEDDEMON")) return "SPEEDDEMON";
        if (cfg.isOn("METALGLDRV_TURBOHAX")) return "TURBOHAX";
        return "Default";
    }

    private static void setProfile(ConfigFile cfg, String v) {
        cfg.set("METALGLDRV_STRICT", v.equals("Strict") ? "1" : v.equals("Super strict") ? "2" : null);
        cfg.set("METALGLDRV_TURBOHAX", v.equals("TURBOHAX") ? "1" : null);
        cfg.set("METALGLDRV_SPEEDDEMON", v.equals("SPEEDDEMON") ? "1" : null);
    }

    private static String currentScale(ConfigFile cfg) {
        String v = cfg.get("METALGLDRV_RENDER_SCALE");
        if (v == null) return "100%";
        try {
            double d = Double.parseDouble(v);
            String best = "100%";
            double bestDiff = Double.MAX_VALUE;
            for (String s : SCALES) {
                double diff = Math.abs(Double.parseDouble(scaleValue(s)) - d);
                if (diff < bestDiff) {
                    bestDiff = diff;
                    best = s;
                }
            }
            return best;
        } catch (NumberFormatException e) {
            return "100%";
        }
    }

    private static String scaleValue(String percent) {
        return String.valueOf(Integer.parseInt(percent.replace("%", "")) / 100.0);
    }

    private static String currentShaderOs(ConfigFile cfg) {
        String v = ConfigFile.lower(cfg.get("spoof_shader_os"));
        if (v == null || v.isEmpty() || v.equals("windows")) return "Windows";
        if (v.equals("linux")) return "Linux";
        if (v.equals("unknown")) return "Unknown";
        return "macOS (no spoofing)";
    }

    private static String shaderOsValue(String label) {
        switch (label) {
            case "Linux": return "linux";
            case "Unknown": return "unknown";
            case "Windows": return "windows";
            default: return "off";
        }
    }

    // ---- Cloth Config through reflection ----

    private static Object toggle(Object entries, String name, boolean value, boolean def, Consumer<Boolean> save, boolean restart,
                                 String... tooltip) throws Exception {
        Object b = call(entries, "startBooleanToggle", text(name), value);
        call(b, "setDefaultValue", def);
        call(b, "setSaveConsumer", save);
        finish(b, restart, tooltip);
        return call(b, "build");
    }

    private static Object selector(Object entries, String name, String[] values, String value, String def, boolean restart,
                                   Consumer<String> save, String... tooltip) throws Exception {
        Object b = call(entries, "startSelector", text(name), values, value);
        call(b, "setDefaultValue", (Object) def);
        call(b, "setSaveConsumer", save);
        Function<Object, Object> names = v -> text(String.valueOf(v));
        call(b, "setNameProvider", names);
        finish(b, restart, tooltip);
        return call(b, "build");
    }

    private static void finish(Object builder, boolean restart, String... tooltip) throws Exception {
        if (restart) {
            try {
                call(builder, "requireRestart", true);  // Cloth Config 4.11+
            } catch (NoSuchMethodException ignored) {
            }
        }
        if (tooltip.length > 0) {
            Object lines = Array.newInstance(componentClass(), tooltip.length + (restart ? 1 : 0));
            for (int i = 0; i < tooltip.length; i++) Array.set(lines, i, text(tooltip[i]));
            if (restart) Array.set(lines, tooltip.length, text("Takes effect when Minecraft restarts."));
            call(builder, "setTooltip", lines);
        }
    }

    private static void add(Object category, Object entry) throws Exception {
        call(category, "addEntry", entry);
    }

    // Calls the public method `name` whose parameters accept `args` (primitive booleans included).
    private static Object call(Object target, String name, Object... args) throws Exception {
        for (Method m : target.getClass().getMethods()) {
            if (!m.getName().equals(name) || m.getParameterCount() != args.length) continue;
            Class<?>[] p = m.getParameterTypes();
            boolean ok = true;
            for (int i = 0; i < p.length && ok; i++) {
                Object a = args[i];
                if (p[i] == boolean.class) ok = a instanceof Boolean;
                else ok = a == null ? !p[i].isPrimitive() : p[i].isInstance(a);
            }
            if (!ok) continue;
            m.setAccessible(true);  // the builders are often package-private implementation classes
            return m.invoke(target, args);
        }
        throw new NoSuchMethodException(target.getClass().getName() + "." + name + " (" + args.length + " arguments)");
    }

    // ---- Minecraft's text class: Mojang names from 26.1, intermediary names before ----

    private static volatile Method literal;

    private static Class<?> componentClass() throws Exception {
        literalMethod();
        return literal.getDeclaringClass();
    }

    static Object text(String s) {
        try {
            return literalMethod().invoke(null, s);
        } catch (Exception e) {
            throw new IllegalStateException("can't create Minecraft text: " + e, e);
        }
    }

    private static Method literalMethod() throws Exception {
        Method m = literal;
        if (m != null) return m;
        ClassLoader cl = ClothConfigScreen.class.getClassLoader();
        try {
            m = Class.forName("net.minecraft.network.chat.Component", false, cl).getMethod("literal", String.class);
        } catch (ClassNotFoundException e) {
            MappingResolver r = FabricLoader.getInstance().getMappingResolver();
            String cls = r.mapClassName("intermediary", "net.minecraft.class_2561");
            String method = r.mapMethodName("intermediary", "net.minecraft.class_2561", "method_43470",
                                            "(Ljava/lang/String;)Lnet/minecraft/class_5250;");
            m = Class.forName(cls, false, cl).getMethod(method, String.class);
        }
        literal = m;
        return m;
    }
}
