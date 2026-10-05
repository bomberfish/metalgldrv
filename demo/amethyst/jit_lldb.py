"""lldb helper that enables JIT for Amethyst on iOS 26+ (TXM) the way StikDebug's
UniversalJIT26.js does, so the launcher can run unattended from a Mac.

Amethyst issues "JIT syscalls" as `brk #0xf00d` with the command in x16:
  0 = detach, 1 = prepare region (x0 = addr or 0 to allocate, x1 = length),
  2 = send script (ignored here).
Region preparation: allocate RX memory through debugserver if needed, then
touch every page through the debugger (read + write back), which is what
lets the process create its writable mirror under TXM.

Usage (see run_game.sh):
  xcrun lldb -o "command script import jit_lldb.py" \
             -o "device select <udid>" -o "device process attach --pid <pid>" \
             -o "jit_loop"
"""
import lldb
import re
import time

PAGE = 0x4000


def _packet(debugger, text):
    ret = lldb.SBCommandReturnObject()
    debugger.GetCommandInterpreter().HandleCommand('process plugin packet send "%s"' % text, ret)
    out = ret.GetOutput() or ""
    m = re.search(r"response:\s*(\S*)", out)
    return m.group(1) if m else ""


def _reg(frame, name):
    return int(frame.FindRegister(name).GetValue(), 16)


def _set_reg(frame, name, value):
    frame.FindRegister(name).SetValueFromCString("0x%x" % value)


def _handle_brk(debugger, process, thread):
    frame = thread.GetFrameAtIndex(0)
    pc = frame.GetPC()
    err = lldb.SBError()
    insn = process.ReadUnsignedFromMemory(pc, 4, err)
    if err.Fail() or (insn & 0xFFE0001F) != 0xD4200000:
        return False
    imm = (insn >> 5) & 0xFFFF
    if imm != 0xF00D:
        return False
    x16, x0, x1 = _reg(frame, "x16"), _reg(frame, "x0"), _reg(frame, "x1")
    _set_reg(frame, "pc", pc + 4)
    if x16 == 0:
        print("[jit] detach requested")
        return "detach"
    if x16 == 1:
        if x0 == 0 and x1 == 0:
            return True
        addr = x0
        if addr == 0:
            resp = _packet(debugger, "_M%x,rx" % x1)
            try:
                addr = int(resp, 16)
            except ValueError:
                print("[jit] RX allocation failed: %r" % resp)
                return True
        # Touch every page via the debugger.
        p = addr & ~(PAGE - 1)
        end = addr + x1
        n = 0
        while p < end:
            b = process.ReadMemory(p, 1, err)
            if err.Success():
                process.WriteMemory(p, b, err)
            n += 1
            p += PAGE
        _set_reg(frame, "x0", addr)
        print("[jit] prepared region 0x%x (+0x%x, %d pages)" % (addr, x1, n))
        return True
    if x16 == 2:
        return True  # extension script: not needed
    print("[jit] unknown command %d" % x16)
    return True


def jit_loop(debugger, command, result, internal_dict):
    debugger.SetAsync(False)
    target = debugger.GetSelectedTarget()
    process = target.GetProcess()
    debugger.HandleCommand("process handle SIGUSR1 SIGUSR2 SIGSEGV SIGBUS SIGPIPE SIGILL -n false -p true -s false")
    print("[jit] handling JIT requests for pid %d" % process.GetProcessID())
    while True:
        process.Continue()
        state = process.GetState()
        if state in (lldb.eStateExited, lldb.eStateDetached, lldb.eStateCrashed):
            print("[jit] process state %d, exit status %d" % (state, process.GetExitStatus()))
            break
        if state != lldb.eStateStopped:
            continue
        handled = False
        for t in process:
            if t.GetStopReason() in (lldb.eStopReasonException, lldb.eStopReasonBreakpoint, lldb.eStopReasonSignal):
                r = _handle_brk(debugger, process, t)
                if r == "detach":
                    process.Detach()
                    print("[jit] detached")
                    return
                if r:
                    handled = True
        if not handled:
            # A real crash or unrelated stop: show it and keep going (signals are passed through).
            for t in process:
                if t.GetStopReason() not in (lldb.eStopReasonNone, lldb.eStopReasonInvalid):
                    print("[jit] stop: thread %d reason %s" % (t.GetIndexID(), t.GetStopDescription(256)))
                    debugger.HandleCommand("thread backtrace %d" % t.GetIndexID())
            time.sleep(0.01)


def __lldb_init_module(debugger, internal_dict):
    debugger.HandleCommand("command script add -f jit_lldb.jit_loop jit_loop")
