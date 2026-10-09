package ca.bomberfish.metalgldrv.fabric;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Line-based editing of config/metalgldrv.properties for the config screen: values are read from and written to
 * their own `key=value` lines, so comments and the rest of the file stay as the user left them.
 */
final class ConfigFile {
    private final Path file;
    private final List<String> lines = new ArrayList<>();

    ConfigFile(Path file) {
        this.file = file;
        try {
            if (Files.exists(file)) lines.addAll(Files.readAllLines(file, StandardCharsets.UTF_8));
        } catch (IOException e) {
            Log.warn("couldn't read " + file + " (" + e + ")");
        }
    }

    // Index of the active (uncommented) line setting `key` (case-insensitive), or -1.
    private int find(String key) {
        for (int i = 0; i < lines.size(); i++) {
            String l = lines.get(i).trim();
            if (l.isEmpty() || l.startsWith("#") || l.startsWith("!")) continue;
            int eq = l.indexOf('=');
            if (eq > 0 && l.substring(0, eq).trim().equalsIgnoreCase(key)) return i;
        }
        return -1;
    }

    /** The value of `key`, or null when it isn't set. */
    String get(String key) {
        int i = find(key);
        if (i < 0) return null;
        String l = lines.get(i);
        return l.substring(l.indexOf('=') + 1).trim();
    }

    boolean isOn(String key) {
        String v = get(key);
        return v != null && !v.isEmpty() && !Bootstrap.isFalse(v);
    }

    /** Sets `key` (replacing its line, else appending one); null removes the line. */
    void set(String key, String value) {
        int i = find(key);
        if (value == null) {
            if (i >= 0) lines.remove(i);
            return;
        }
        String line = key + "=" + value;
        if (i >= 0) lines.set(i, line);
        else lines.add(line);
    }

    void save() {
        try {
            Files.createDirectories(file.getParent());
            Files.write(file, lines, StandardCharsets.UTF_8);
            Log.info("saved " + file);
        } catch (IOException e) {
            Log.warn("couldn't write " + file + " (" + e + ")");
        }
    }

    static String lower(String s) {
        return s == null ? null : s.trim().toLowerCase(Locale.ROOT);
    }
}
