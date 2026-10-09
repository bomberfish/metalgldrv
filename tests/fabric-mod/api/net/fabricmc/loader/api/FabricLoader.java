package net.fabricmc.loader.api;

import java.nio.file.Path;

/** Stand-in for Fabric Loader's FabricLoader: only what the mod calls. */
public interface FabricLoader {
    static FabricLoader getInstance() {
        return FakeLoader.instance;
    }

    Path getConfigDir();

    boolean isModLoaded(String id);
}
