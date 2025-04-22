.section __DATA, __data
my_constant: .double 12.5

.section __TEXT, __text
.globl _main
.p2align 2
_main:
    adrp x9,  my_constant@PAGE
    add x9, x9, my_constant@PAGEOFF
    ldr d0, [x9]

    fmov d1, #0.0
    fcmp d0, d1
    b.ge .positive

    fneg d0, d0

.positive:
    fcvtzs w0, d0
    ret