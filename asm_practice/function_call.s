.section __DATA, __data
my_constant: .double -7.5

.section __TEXT, __text
.globl _main
.extern _fabs
.p2align 2
_main:
    stp x29, x30, [sp, #-16]!
    mov x29, sp

    adrp x9, my_constant@PAGE
    add  x9, x9, my_constant@PAGEOFF
    ldr  d0, [x9]

    bl _fabs

    fcvtzs w0, d0

    ldp x29, x30, [sp], #16
    ret