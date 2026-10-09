package net.fabricmc.loader.api;

import java.nio.file.Path;
import java.util.Set;

/** The harness's FabricLoader: a config dir and a set of "loaded" mod ids. */
public final class FakeLoader implements FabricLoader {
    static FabricLoader instance;

    private final Path configDir;
    private final Set<String> mods;

    public FakeLoader(Path configDir, Set<String> mods) {
        this.configDir = configDir;
        this.mods = mods;
    }

    public static void install(FakeLoader l) {
        instance = l;
    }

    @Override
    public Path getConfigDir() {
        return configDir;
    }

    @Override
    public boolean isModLoaded(String id) {
        return mods.contains(id);
    }
}
