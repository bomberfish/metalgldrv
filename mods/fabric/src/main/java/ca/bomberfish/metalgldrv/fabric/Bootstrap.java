package ca.bomberfish.metalgldrv.fabric;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.Reader;
import java.io.Writer;
import java.lang.reflect.Method;
import java.net.JarURLConnection;
import java.net.URL;
import java.net.URLConnection;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Properties;
import java.util.Set;
import java.util.function.Predicate;
import java.util.jar.Manifest;

/**
 * What the preLaunch entrypoint does, without Fabric types (the test harness in
 * tests/fabric-mod drives it through the entrypoint with a stand-in loader):
 * read the config, look for conflicts, extract the bundled universal dylib to
 * ~/Library/Caches/metalgldrv/&lt;sha256 prefix&gt;/ and System.load it. The driver's
 * JNI_OnLoad then installs its NSOpenGL/CGL hooks and sets org.lwjgl.opengl.libname,
 * and reports back in the metalgldrv.activation / metalgldrv.version properties.
 */
final class Bootstrap {
    static final String CONFIG_FILE = "metalgldrv.properties";
    static final String RESOURCE_DIR = "/natives/macos/";
    static final String DYLIB = "libmetalgldrv.dylib";
    static final String GL_LIBNAME = "org.lwjgl.opengl.libname";
    static final String[] LWJGL_LIBNAME_KEYS = {GL_LIBNAME, "org.lwjgl.glfw.libname", "org.lwjgl.egl.libname",
                                                "org.lwjgl.sdl.libname"};
    // LWJGL's windowing bindings: GLFW, or SDL 3 from Minecraft 26.3 on.
    static final String[] WINDOW_CLASSES = {"org.lwjgl.glfw.GLFW", "org.lwjgl.sdl.SDL"};
    // Mods that replace or take over Minecraft's renderer / GL backend.
    static final String[] CONFLICTING_MODS = {"vulkanmod"};

    private final Path configDir;
    private final Predicate<String> isModLoaded;

    Bootstrap(Path configDir, Predicate<String> isModLoaded) {
        this.configDir = configDir;
        this.isModLoaded = isModLoaded;
    }

    void run() throws Exception {
        Properties build = buildInfo();
        String banner = "metalgldrv mod " + build.getProperty("modVersion", "?") + " (driver " + build.getProperty("driverVersion", "?") + ")";
        String os = System.getProperty("os.name", "");
        if (!os.toLowerCase(Locale.ROOT).startsWith("mac")) {
            Log.info(banner + ": not macOS (" + os + "), nothing to do");
            return;
        }
        Properties cfg = loadConfig();
        String enabled = System.getProperty("metalgldrv.enabled");
        boolean fromJvm = enabled != null;
        if (!fromJvm) enabled = cfg.getProperty("enabled", "true");
        if (isFalse(enabled)) {
            Log.info(banner + ": disabled by " + (fromJvm ? "-Dmetalgldrv.enabled" : "enabled=" + enabled.trim() + " in config/" + CONFIG_FILE)
                     + "; Minecraft uses Apple's OpenGL");
            return;
        }
        String lwjgl = lwjglVersion();
        Log.info(banner + ": activating (LWJGL " + (lwjgl != null ? lwjgl : "version unknown") + ")");

        String activation = System.getProperty("metalgldrv.activation");
        if ("agent".equals(activation) || "jni".equals(activation)) {
            Log.info("metalgldrv is already active (" + activation + ", " + System.getProperty("metalgldrv.library") + "); not loading the bundled copy");
            return;
        }

        boolean force = isTrue(System.getProperty("metalgldrv.force", cfg.getProperty("force", "false")));
        List<String> conflicts = new ArrayList<>();
        for (String key : LWJGL_LIBNAME_KEYS) {
            String v = System.getProperty(key);
            if (v != null) conflicts.add("-D" + key + "=" + v);
        }
        for (String id : CONFLICTING_MODS)
            if (isModLoaded.test(id)) conflicts.add("mod '" + id + "'");
        if (LWJGL2.equals(lwjgl)) conflicts.add("LWJGL 2 (only LWJGL 3 is supported)");
        String replacedLibname = null;
        if (!conflicts.isEmpty()) {
            if (!force) {
                Log.warn("not activating: found " + String.join(", ", conflicts) + ". Remove the JVM arguments / mod, or set force=true in config/"
                         + CONFIG_FILE + ". Minecraft uses " + (System.getProperty(GL_LIBNAME) != null ? "the configured GL library" : "Apple's OpenGL"));
                return;
            }
            Log.warn("force=true: activating despite " + String.join(", ", conflicts));
            replacedLibname = System.getProperty(GL_LIBNAME);
            if (replacedLibname != null) {
                Log.warn("force=true: replacing " + GL_LIBNAME + "=" + replacedLibname);
                System.clearProperty(GL_LIBNAME);
            }
        }

        String loaded = windowingLoaded();
        if (loaded != null)
            Log.warn(loaded + " is already loaded; a window with Apple's OpenGL may already exist and keep it");

        passThroughSettings(cfg);

        try {
            String custom = cfg.getProperty("library", "").trim();
            // The real path: the driver reports itself (dladdr) by it.
            Path lib = (custom.isEmpty() ? extract(build) : Paths.get(custom)).toRealPath();
            if (!custom.isEmpty()) Log.info("using library=" + lib + " from config/" + CONFIG_FILE + " instead of the bundled driver");
            System.load(lib.toString());

            activation = System.getProperty("metalgldrv.activation");
            String version = System.getProperty("metalgldrv.version", "?");
            if ("jni".equals(activation)) {
                Log.info("metalgldrv " + version + " active: " + lib + "; " + GL_LIBNAME + "=" + System.getProperty(GL_LIBNAME));
            } else if (activation == null) {
                Log.warn("loaded " + lib + " but it did not report back (no JNI_OnLoad? an older driver?); Minecraft may use Apple's OpenGL");
            } else {
                Log.warn("metalgldrv " + version + " not activated (" + activation + "); Minecraft uses "
                         + (System.getProperty(GL_LIBNAME) != null || replacedLibname != null ? "the configured GL library" : "Apple's OpenGL"));
            }
        } finally {
            // Not active after all: put back what force=true took away.
            if (replacedLibname != null && !"jni".equals(System.getProperty("metalgldrv.activation")) && System.getProperty(GL_LIBNAME) == null)
                System.setProperty(GL_LIBNAME, replacedLibname);
        }
    }

    // ---- config

    private static final String DEFAULT_CONFIG = String.join("\n",
        "# metalgldrv: runs Minecraft's OpenGL on metalgldrv (OpenGL 4.6 on Metal 4).",
        "# Read once per launch, before the game window opens.",
        "#",
        "# enabled=false turns the mod off (Minecraft then uses Apple's OpenGL).",
        "# The JVM argument -Dmetalgldrv.enabled=false|true overrides this.",
        "enabled=true",
        "#",
        "# The mod doesn't activate when it finds a conflict: -Dorg.lwjgl.opengl.libname,",
        "# -Dorg.lwjgl.glfw.libname, -Dorg.lwjgl.egl.libname or -Dorg.lwjgl.sdl.libname JVM",
        "# arguments (an old Mesa or custom GLFW setup, Prism's \"Use system installation of",
        "# GLFW\") or a renderer mod such as VulkanMod. force=true activates anyway",
        "# and points org.lwjgl.opengl.libname at metalgldrv.",
        "force=false",
        "#",
        "# library=/path/to/libmetalgldrv.dylib loads that driver instead of the bundled one.",
        "#library=",
        "#",
        "# Driver settings: any METALGLDRV_<NAME> environment variable of the driver, as",
        "# METALGLDRV_<NAME>=value (or metalgldrv.<name>=value). Environment variables and",
        "# -Dmetalgldrv.<name> JVM arguments that are already set win. For example:",
        "#METALGLDRV_LOG=3",
        "#METALGLDRV_LOG_FILE=documents",
        "#METALGLDRV_LOG_UNSUPPORTED=1",
        "");

    private Properties loadConfig() {
        Properties p = new Properties();
        Path file = configDir.resolve(CONFIG_FILE);
        try {
            if (!Files.exists(file)) {
                Files.createDirectories(configDir);
                try (Writer w = Files.newBufferedWriter(file, StandardCharsets.UTF_8)) {
                    w.write(DEFAULT_CONFIG);
                }
                Log.info("wrote default config " + file);
            }
            try (Reader r = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
                p.load(r);
            }
        } catch (Exception e) {
            Log.warn("couldn't read " + file + " (" + e + "); using defaults");
        }
        return p;
    }

    // METALGLDRV_<NAME> / metalgldrv.<name> config entries become metalgldrv.<name> system
    // properties (unless set), which the driver's JNI_OnLoad copies to the environment.
    private static void passThroughSettings(Properties cfg) {
        for (String key : cfg.stringPropertyNames()) {
            String prop;
            if (key.regionMatches(true, 0, "METALGLDRV_", 0, 11)) prop = "metalgldrv." + key.substring(11);
            else if (key.regionMatches(true, 0, "metalgldrv.", 0, 11)) prop = "metalgldrv." + key.substring(11);
            else continue;
            if (prop.length() == 11) continue;
            String value = cfg.getProperty(key).trim();
            if (System.getProperty(prop) == null) System.setProperty(prop, value);
        }
    }

    static boolean isFalse(String v) {
        v = v.trim();
        return v.equals("0") || v.equalsIgnoreCase("false") || v.equalsIgnoreCase("no") || v.equalsIgnoreCase("off");
    }

    static boolean isTrue(String v) {
        v = v.trim();
        return v.equals("1") || v.equalsIgnoreCase("true") || v.equalsIgnoreCase("yes") || v.equalsIgnoreCase("on");
    }

    // ---- LWJGL

    static final String LWJGL2 = "2";

    /**
     * LWJGL 3's version from its jar manifest (e.g. "3.4.3 (build 4)"), "2" for LWJGL 2, or null.
     * Looks at class files as resources only: nothing is loaded.
     */
    static String lwjglVersion() {
        ClassLoader cl = Thread.currentThread().getContextClassLoader();
        if (cl == null) cl = Bootstrap.class.getClassLoader();
        try {
            URL url = cl.getResource("org/lwjgl/system/Library.class");
            if (url == null) return cl.getResource("org/lwjgl/Sys.class") != null ? LWJGL2 : null;
            URLConnection c = url.openConnection();
            if (c instanceof JarURLConnection) {
                Manifest mf = ((JarURLConnection) c).getManifest();
                String spec = mf != null ? mf.getMainAttributes().getValue("Specification-Version") : null;
                String impl = mf != null ? mf.getMainAttributes().getValue("Implementation-Version") : null;
                if (spec != null) return impl != null ? spec + " (" + impl + ")" : spec;
            }
            return "3";
        } catch (Throwable t) {
            return null;
        }
    }

    // ---- GLFW state

    /** "class (loader)" when LWJGL's GLFW or SDL class is already loaded, else null. Never loads them. */
    static String windowingLoaded() {
        Set<ClassLoader> loaders = new LinkedHashSet<>();
        loaders.add(Bootstrap.class.getClassLoader());
        loaders.add(Thread.currentThread().getContextClassLoader());
        loaders.add(ClassLoader.getSystemClassLoader());
        for (String name : WINDOW_CLASSES)
            for (ClassLoader cl : loaders)
                if (cl != null && findLoadedClass(cl, name)) return name + " (" + cl.getClass().getName() + ")";
        return null;
    }

    private static boolean findLoadedClass(ClassLoader cl, String name) {
        // Knot (Fabric's game class loader) exposes findLoadedClassFwd; the JDK's
        // protected findLoadedClass is only reachable with --add-opens java.base/java.lang.
        for (Class<?> c = cl.getClass(); c != null; c = c.getSuperclass()) {
            for (String m : new String[] {"findLoadedClassFwd", "findLoadedClass"}) {
                try {
                    Method method = c.getDeclaredMethod(m, String.class);
                    method.setAccessible(true);
                    return method.invoke(cl, name) != null;
                } catch (NoSuchMethodException e) {
                    // try the next one
                } catch (Throwable e) {
                    break;  // not accessible: unknown
                }
            }
        }
        return false;
    }

    // ---- the bundled driver

    static Properties buildInfo() {
        Properties p = new Properties();
        try (InputStream in = Bootstrap.class.getResourceAsStream("/metalgldrv-build.properties")) {
            if (in != null) p.load(in);
        } catch (IOException ignored) {
            // versions show as "?"
        }
        return p;
    }

    /** Extracts the bundled dylib to ~/Library/Caches/metalgldrv/&lt;sha256 prefix&gt;/ (reused when intact). */
    static Path extract(Properties build) throws Exception {
        String sha = build.getProperty("sha256", "").trim();
        if (sha.length() != 64) throw new IOException("metalgldrv-build.properties has no sha256 for the bundled dylib");
        Path dir = Paths.get(System.getProperty("user.home"), "Library", "Caches", "metalgldrv", sha.substring(0, 16));
        Path target = dir.resolve(DYLIB);
        if (Files.isRegularFile(target) && sha.equals(sha256(target))) {
            Log.info("using cached " + target);
            return target;
        }
        Files.createDirectories(dir);
        Path tmp = Files.createTempFile(dir, DYLIB, ".tmp");
        try {
            try (InputStream in = Bootstrap.class.getResourceAsStream(RESOURCE_DIR + DYLIB)) {
                if (in == null) throw new IOException("the jar has no " + RESOURCE_DIR + DYLIB);
                try (OutputStream out = Files.newOutputStream(tmp)) {
                    in.transferTo(out);
                }
            }
            String got = sha256(tmp);
            if (!sha.equals(got)) throw new IOException("bundled dylib has sha256 " + got + ", expected " + sha);
            // A new inode, renamed into place: never rewrite a dylib another process may
            // have mapped (that invalidates its code signature pages).
            Files.move(tmp, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } finally {
            Files.deleteIfExists(tmp);
        }
        Log.info("extracted the driver to " + target);
        return target;
    }

    static String sha256(Path file) throws Exception {
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        try (InputStream in = Files.newInputStream(file)) {
            byte[] buf = new byte[1 << 16];
            for (int n; (n = in.read(buf)) > 0;) md.update(buf, 0, n);
        }
        StringBuilder sb = new StringBuilder();
        for (byte b : md.digest()) sb.append(String.format("%02x", b));
        return sb.toString();
    }
}
