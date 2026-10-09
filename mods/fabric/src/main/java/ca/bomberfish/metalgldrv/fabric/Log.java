package ca.bomberfish.metalgldrv.fabric;

/** Logs through SLF4J (which Minecraft and Fabric Loader route to the game log), or stderr without it. */
final class Log {
    private Log() {}

    private interface Sink {
        void info(String msg);
        void warn(String msg, Throwable t);
    }

    private static final Sink SINK = create();

    private static Sink create() {
        try {
            return Slf4j.create();
        } catch (Throwable t) {
            return new Sink() {
                @Override
                public void info(String msg) {
                    System.err.println("[metalgldrv] " + msg);
                }

                @Override
                public void warn(String msg, Throwable t) {
                    System.err.println("[metalgldrv] WARN: " + msg);
                    if (t != null) t.printStackTrace();
                }
            };
        }
    }

    // Separate class: links against SLF4J only when it is there.
    private static final class Slf4j {
        static Sink create() {
            final org.slf4j.Logger l = org.slf4j.LoggerFactory.getLogger("metalgldrv");
            return new Sink() {
                @Override
                public void info(String msg) {
                    l.info(msg);
                }

                @Override
                public void warn(String msg, Throwable t) {
                    if (t != null) l.warn(msg, t);
                    else l.warn(msg);
                }
            };
        }
    }

    static void info(String msg) {
        SINK.info(msg);
    }

    static void warn(String msg) {
        SINK.warn(msg, null);
    }

    static void warn(String msg, Throwable t) {
        SINK.warn(msg, t);
    }
}
