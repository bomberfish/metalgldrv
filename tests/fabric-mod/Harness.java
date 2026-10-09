import java.io.InputStream;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import net.fabricmc.loader.api.FakeLoader;
import net.fabricmc.loader.api.entrypoint.PreLaunchEntrypoint;

/**
 * Runs the mod's preLaunch entrypoint (named by the jar's fabric.mod.json) under a
 * stand-in FabricLoader, then prints what a test checks as "key=value" lines.
 *
 *   -Dtest.configDir=DIR      config dir handed to the mod
 *   -Dtest.mods=a,b           mod ids that count as loaded
 *   -Dtest.preloadGlfw=true   load (without initializing) a class named org.lwjgl.glfw.GLFW first
 *   -Dtest.hideAgent=true     clear metalgldrv.activation (set by an -agentpath copy) before the mod runs
 *   -Dtest.envProbe=LIB       JNI library with Harness.getenv0 (reads the C environment)
 *   -Dtest.env=A,B            environment variables to print through the probe
 */
public final class Harness {
    static native String getenv0(String name);

    public static void main(String[] args) throws Exception {
        Path configDir = Paths.get(System.getProperty("test.configDir"));
        Set<String> mods = new HashSet<>();
        String m = System.getProperty("test.mods", "");
        if (!m.isEmpty()) mods.addAll(Arrays.asList(m.split(",")));
        FakeLoader.install(new FakeLoader(configDir, mods));
        if (Boolean.getBoolean("test.hideAgent")) System.clearProperty("metalgldrv.activation");
        if (Boolean.getBoolean("test.preloadGlfw")) Class.forName("org.lwjgl.glfw.GLFW", false, Harness.class.getClassLoader());

        String json;
        URL url = Harness.class.getResource("/fabric.mod.json");
        try (InputStream in = url.openStream()) {
            json = new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
        Matcher e = Pattern.compile("\"preLaunch\"\\s*:\\s*\\[\\s*\"([^\"]+)\"").matcher(json);
        if (!e.find()) throw new IllegalStateException("no preLaunch entrypoint in fabric.mod.json");
        PreLaunchEntrypoint ep = (PreLaunchEntrypoint) Class.forName(e.group(1)).getDeclaredConstructor().newInstance();
        ep.onPreLaunch();

        for (String k : new String[] {"metalgldrv.activation", "metalgldrv.version", "metalgldrv.library", "org.lwjgl.opengl.libname"})
            System.out.println("prop " + k + "=" + System.getProperty(k));
        // What the Iris mixin would report instead of MC_OS_MAC (null: no spoof).
        System.out.println("shaderos=" + Class.forName("ca.bomberfish.metalgldrv.fabric.ShaderOs").getMethod("define").invoke(null));
        String probe = System.getProperty("test.envProbe");
        if (probe != null) {
            System.load(probe);
            for (String var : System.getProperty("test.env", "").split(","))
                if (!var.isEmpty()) System.out.println("env " + var + "=" + getenv0(var));
        }
    }
}
