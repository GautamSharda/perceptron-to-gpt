.section __DATA, __data
constant_one: .double 20.0
constant_two: .double 10.0

.section __TEXT, __text
.globl _main
.p2align 2
_main:
    adrp x9, constant_one@PAGE
    add x9, x9, constant_one@PAGEOFF
    ldr d0, [x9]

    adrp x10, constant_two@PAGE
    add x10, x10, constant_two@PAGEOFF
    ldr d1, [x10]

    fadd d2, d0, d1
    fsub d3, d0, d1
    fmul d4, d2, d3

    fcvtzs w0, d4
    ret
