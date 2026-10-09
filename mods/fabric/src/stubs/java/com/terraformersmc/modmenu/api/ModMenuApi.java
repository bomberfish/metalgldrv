package com.terraformersmc.modmenu.api;

// Compile-time stub of Mod Menu's API (the subset the mod uses). The real interface comes from Mod Menu at
// run time; this stub keeps Minecraft classes out of the compile (the mod has no Minecraft dependency).
public interface ModMenuApi {
    default ConfigScreenFactory<?> getModConfigScreenFactory() {
        return null;
    }
}
