.section __DATA, __data
weight: .double 20.0
bias: .double 10.0
x: .double 2.0
y: .double 2.0
learning_rate: .double 0.025

.section __TEXT, __text
.extern _fabs
.globl _main
.p2align 2
_main:
    // store return memory address & frame pointer in stack, move stack pointer
    stp x29, x30, [sp, #-16]!
    mov x29, sp

    // load constants
    adrp x9, weight@PAGE
    add x9, x9, weight@PAGEOFF
    ldr d0, [x9]

    adrp x10, bias@PAGE
    add x10, x10, bias@PAGEOFF
    ldr d1, [x10]
    fmov d13, d1

    adrp x11, x@PAGE
    add x11, x11, x@PAGEOFF
    ldr d2, [x11]

    adrp x12, y@PAGE
    add x12, x12, y@PAGEOFF
    ldr d3, [x12]

    adrp x13, learning_rate@PAGE
    add x13, x13, learning_rate@PAGEOFF
    ldr d4, [x13]

    // forward pass
    fmul d5, d0, d2
    fadd d6, d5, d13

    // loss
    fsub d7, d6, d3
    fmov d0, d7
    bl _fabs

    // diff
    fmov d8, #0.0
    fcmp d0, d8
    b.lt .negative
    
    ldr d0, [x9]
    fmov d9, #1.0
    b .abs_diff_done

.negative:
    fmov d9, #-1.0

.abs_diff_done:
    fmov d12, #1.0
    fmul d10, d0, d9
    fmul d11, d12, d9

    fmul d10, d10, d4
    fsub d0, d0, d10

    fmul d11, d11, d4
    fsub d1, d13, d11

    fcvtzs w0, d0

    ldp x29, x30, [sp], #16
    ret
    







