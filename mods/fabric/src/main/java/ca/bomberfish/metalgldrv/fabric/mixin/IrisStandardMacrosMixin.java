package ca.bomberfish.metalgldrv.fabric.mixin;

import ca.bomberfish.metalgldrv.fabric.ShaderOs;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Pseudo;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/**
 * Iris's StandardMacros.getOsString() returns the MC_OS_* define for shader packs (both the GLSL
 * defines and shaders.properties conditionals come from it). @Pseudo: Iris is optional, and both its
 * current (net.irisshaders) and pre-1.7 (net.coderbot) packages are targeted; neither is obfuscated.
 */
@Pseudo
@Mixin(targets = {"net.irisshaders.iris.gl.shader.StandardMacros", "net.coderbot.iris.gl.shader.StandardMacros"}, remap = false)
public abstract class IrisStandardMacrosMixin {
    @Inject(method = "getOsString", at = @At("HEAD"), cancellable = true, remap = false, require = 0)
    private static void metalgldrv$shaderOs(CallbackInfoReturnable<String> cir) {
        String define = ShaderOs.define();
        if (define != null) cir.setReturnValue(define);
    }
}
