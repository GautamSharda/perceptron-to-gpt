.section __DATA, __data
my_constant: .double 3.14

.section __TEXT, __text
.globl _main
.p2align 2
_main:
    adrp x9, my_constant@PAGE
    add x9, x9, my_constant@PAGEOFF
    ldr d0, [x9]
    fcvtzs w0, d0
    ret