package ca.bomberfish.metalgldrv.fabric;

import net.fabricmc.loader.api.FabricLoader;
import net.fabricmc.loader.api.entrypoint.PreLaunchEntrypoint;

/**
 * Fabric preLaunch entrypoint: runs in Knot before Minecraft's main(), so before GLFW
 * creates the window. Never throws: on any failure the game keeps Apple's OpenGL.
 */
public final class MetalGLDrvPreLaunch implements PreLaunchEntrypoint {
    @Override
    public void onPreLaunch() {
        try {
            FabricLoader loader = FabricLoader.getInstance();
            new Bootstrap(loader.getConfigDir(), loader::isModLoaded).run();
        } catch (Throwable t) {
            try {
                Log.warn("activation failed; Minecraft keeps Apple's OpenGL", t);
            } catch (Throwable ignored) {
                // nothing left to report with
            }
        }
    }
}
